import os
import pickle
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from .constants_val import *
from .utils import get_imgs, get_tokenizer, read_from_dicom, check_element_type
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
from copy import deepcopy
import random
from memory_profiler import profile
from .transforms import DataTransforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



class CheXpertPretrainingDataset(data.Dataset):
    def __init__(self, split='train', transform=None, data_pct=1.0,
                 imsize=256, max_words=64, sent_num=3, masked_lm_ratio=0,
                 llm_type="gpt", negative_prompt=False, prompt_ft=False,
                 cls_prompt=False, five_cls=False,
                 train_sub_set=False, **kwargs):
        super().__init__()
        if not os.path.exists(CHEXPERT_DATA_DIR):
            raise RuntimeError(f"{CHEXPERT_DATA_DIR} does not exist!")
        
        self.llm_type = llm_type
        self.transform = transform
        self.imsize = imsize
        self.masked_lm_ratio = masked_lm_ratio
        self.split = split
        self.prompt_ft = prompt_ft
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        self.cls_prompt = cls_prompt
        self.five_cls = five_cls
        if split == 'train':
            # Use CheXbert as recommended by the authors
            train_csv = CHEXBERT_TRAIN_5_CSV if five_cls else CHEXBERT_TRAIN_CSV
            print(f"### Using {train_csv} for training")
            self.df = pd.read_csv(train_csv)
        elif split == 'valid':
            valid_csv = CHEXPERT_VALID_5_CSV if five_cls else CHEXPERT_VALID_CSV
            print(f"### Using {valid_csv} for validation")
            self.df = pd.read_csv(valid_csv)
        elif split == 'test':
            self.df = pd.read_csv(CHEXPERT_TEST_CSV)
            self.cls_prompt = True
        else:
            raise ValueError(f"split {split} not supported")
        self.df = self.df[self.df[CHEXPERT_VIEW_COL].isin(["PA", "AP"])]
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:])))
        
        sub_train_set = split == 'train' or prompt_ft or train_sub_set
        if data_pct != 1.0 and sub_train_set:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        # load studies and study to text mapping
        if negative_prompt:
            max_words = 96
        self.filenames, self.path2sent = self.load_text_data(split, negative_prompt)
        
        self.tokenizer = get_tokenizer(llm_type)
        self.max_words = max_words

    def load_text_data(self, split, negative_prompt=False):
        base_filename = f"{split}_captions.pickle"
        if negative_prompt:
            base_filename = base_filename.replace(".pickle", "_negative.pickle")
        if self.llm_type != 'gpt':
            base_filename = base_filename.replace(".pickle", f"_{self.llm_type}.pickle")
        if self.prompt_ft:
            base_filename = base_filename.replace(".pickle", "_prompt_ft.pickle")
        filepath = os.path.join(CHEXPERT_DATA_DIR, base_filename)
        
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exist. Creating captions...")
            path2sent = self.create_path_2_sent_mapping(negative_prompt)
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)
        
        # Some of the paths in the data frame are not in the captions
        filenames = []
        # Only store label when doing five_classes classification
        self.path2label = {}
        for idx, row in self.df.iterrows():
            path = row[CHEXPERT_PATH_COL]
            if os.path.join(DATA_BASE_DIR, 'CheXpert', path) in path2sent:
                if self.five_cls:
                    for label_idx, col in enumerate(CHEXPERT_FINDINGS_5):
                        if not np.isnan(row[col]) and int(float(row[col])) == 1:
                            self.path2label[path] = label_idx
                            break
                filenames.append(path)
        if self.five_cls:
            print(Counter(self.path2label.values()))
        return filenames, path2sent
    
    def create_path_2_sent_mapping(self, negative_prompt=False):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than iter tuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # Start with base caption
            captions = '' if self.prompt_ft else CHEXPERT_BASE_CAPTION

            # Add findings
            label_cnt = 0
            for col in CHEXPERT_FINDINGS:
                if row[col] == 1:
                    captions += col
                    captions += " "
                label_cnt += 1

            if negative_prompt:
                if label_cnt == 0:
                    captions += "no finding "
                captions += CHEXPERT_NEGATIVE_CAPTION
                for col in CHEXPERT_FINDINGS:
                    if col == "No Finding":
                        continue
                    if row[col] == 0:
                        captions += col
                        captions += " "
                    label_cnt += 1
            if label_cnt == 0:
                continue

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                if len(tokens) <= 0:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 1:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}] {len(sent_lens)}"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}] {len(sent_lens)}"
        )

        return path2sent
    
    def __len__(self):
        return len(self.filenames)
    
    def random_mask(self, tokens, mask_ratio=0.5):
        # TODO: make this with a probability of 0.3
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1]-1):
            if masked_tokens[0][i] == 0:
                break

            prob = random.random()
            if prob < mask_ratio:
                masked_tokens[0][i] = 103

        return masked_tokens
    
    def get_caption(self, path, series_sents=None):
        if series_sents is None:
            series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)
        if self.prompt_ft:
            sent = "X " + sent

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        masked_ids = self.random_mask(tokens['input_ids'], mask_ratio=self.masked_lm_ratio)
        tokens['masked_ids'] = masked_ids
        if self.prompt_ft:
            # Remove the prefixing "X" and prefix "Ġ" token
            tokens['input_ids'] = tokens['input_ids'][:, 1:]
            tokens['masked_ids'] = tokens['masked_ids'][:, 1:]
            tokens['attention_mask'] = tokens['attention_mask'][:, 1:]

        return tokens, x_len
    
    def get_multi_hot_label(self, index, get_full=False):
        FINDS = CHEXPERT_FINDINGS_5 if self.five_cls else CHEXPERT_FINDINGS
        multi_hot_label = torch.zeros(len(FINDS))
        for i, col in enumerate(FINDS):
            if self.df.iloc[index][col] == 1:
                # Positive
                multi_hot_label[i] = 1
            if get_full and self.df.iloc[index][col] == -1:
                # Uncertain
                multi_hot_label[i] = 0
            if get_full and self.df.iloc[index][col] == 0:
                # Negative 
                multi_hot_label[i] = -1
        return multi_hot_label

    def __cls_getitem__(self, index):
        key = self.filenames[index]
        one_hot_label = self.get_multi_hot_label(index)
        if self.zero_shot_caps is None:
            zero_shot_caps = []
            zero_shot_caps_len = []
            FINDS = CHEXPERT_FINDINGS_5 if self.five_cls else CHEXPERT_FINDINGS
            base_caption = '' if self.prompt_ft else CHEXPERT_BASE_CAPTION
            for i in range(len(FINDS)):
                captions = base_caption + FINDS[i]
                captions = captions.replace("\n", " ").lower()
                cap, cap_len = self.get_caption(None, [captions])
                zero_shot_caps.append(cap)
                zero_shot_caps_len.append(cap_len)

            stacked_caps = {}
            for cap in zero_shot_caps:
                for k, v in cap.items():
                    if k not in stacked_caps:
                        stacked_caps[k] = v
                    else:
                        stacked_caps[k] = torch.concat([stacked_caps[k], v], dim=0)
            zero_shot_caps_len = torch.tensor(zero_shot_caps_len)
            self.zero_shot_caps = stacked_caps
            self.zero_shot_caps_len = zero_shot_caps_len
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, self.zero_shot_caps, self.zero_shot_caps_len, key, one_hot_label, imgs

    def __getitem__(self, index):
        if self.cls_prompt:
            return self.__cls_getitem__(index)
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        multi_hot_label = self.get_multi_hot_label(index)
        
        # get unpaired text
        index_list = list(range(len(self.filenames)))
        index_list.remove(index)
        unpaired_index = random.sample(index_list, 1)[0]
        unpaired_caps, unpaired_cap_len = self.get_caption(self.filenames[unpaired_index])
        unpaired_multi_hot_label = self.get_multi_hot_label(unpaired_index)
        return imgs, caps, cap_len, key, multi_hot_label, imgs


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, ids, tokens, attention, masked_ids, labels = [], [], [], [], [], [], []
    ext_imgs = []
    orig_imgs = []
    path = []
    tokens_type_id_exist = False
    eval_mode = False
    for b in batch:
        img, cap, cap_l, p, label, orig_img = b
        if isinstance(img, list):
            img, ext_img = img
            ext_imgs.append(ext_img)
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        if "token_type_ids" in cap:
            tokens.append(cap["token_type_ids"])
            tokens_type_id_exist = True
        labels.append(label)
        attention.append(cap["attention_mask"])
        masked_ids.append(cap["masked_ids"])
        path.append(p)
        orig_imgs.append(orig_img)

    # stack
    imgs = torch.stack(imgs)
    ext_imgs = torch.stack(ext_imgs) if len(ext_imgs) > 0 else None
    orig_imgs = torch.stack(orig_imgs)
    ids = torch.stack(ids).squeeze()
    if tokens_type_id_exist:
        tokens = torch.stack(tokens).squeeze()
    labels = torch.stack(labels).squeeze()
    attention = torch.stack(attention).squeeze()
    masked_ids = torch.stack(masked_ids).squeeze()

    # sort and add to dictionary
    sorted_cap_indices = torch.arange(len(cap_len))
    try:
        sorted_cap_lens = torch.tensor(cap_len)
    except TypeError:
        sorted_cap_lens = torch.stack(cap_len, 0)

    path = np.array(path)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices] if tokens_type_id_exist else None,
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices],
        "masked_ids": masked_ids[sorted_cap_indices],
        "multi_hot_label": labels[sorted_cap_indices],
        "orig_imgs": orig_imgs[sorted_cap_indices],
    }
    if ext_imgs is not None:
        return_dict["ext_imgs"] = ext_imgs
    return return_dict

