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
import random
from memory_profiler import profile
from .transforms import DataTransforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class EmbedPretrainingDataset(data.Dataset):
    def __init__(self, split='train', transform=None, data_pct=1.0,
                 imsize=256, max_words=72, llm_type="gpt", simple_cap=False,
                 cls_prompt=False, train_sub_set=False, structural_cap=False,
                 natural_cap=False, instance_test_cap=False,
                 balanced_test=False,
                 prob_diff_dcm=0.5, small_balanced_train=False,
                 pred_density=False, **kwargs):
        super().__init__()
        if not os.path.exists(EMBED_DATA_DIR):
            raise RuntimeError(f"{EMBED_DATA_DIR} does not exist!")
        
        self.llm_type = llm_type
        self.transform = transform
        self.imsize = imsize
        self.split = split
        self.cls_prompt = cls_prompt
        self.max_words = max_words
        self.structural_cap = structural_cap
        self.simple_cap = simple_cap
        self.natural_cap = natural_cap
        self.instance_test_cap = instance_test_cap
        self.balanced_test = balanced_test
        self.prob_diff_dcm = prob_diff_dcm
        self.small_balanced_train = small_balanced_train
        self.pred_density = pred_density
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        if split == 'train':
            self.df = pd.read_csv(EMBED_TRAIN_META_CSV)
        elif split == 'valid':
            self.df = pd.read_csv(EMBED_VALID_META_CSV)
        elif split == 'test':
            self.df = pd.read_csv(EMBED_TEST_META_CSV)
            self.cls_prompt = True
        else:
            raise ValueError(f"split {split} not supported")
        self.df_anno = pd.read_csv(EMBED_ANNO_CSV_REDUCED)
        self.df_anno_full = pd.read_csv(EMBED_ANNO_CSV)
        df_legends = pd.read_csv(EMBED_LEGENDS_CSV)
        
        self.massshape_dict = {row['Code']: row['Meaning'] for _, row in df_legends[df_legends['Header in export'] == 'massshape'].iterrows()}
        self.massdensity_dict = {row['Code']: row['Meaning'] for _, row in df_legends[df_legends['Header in export'] == 'massdens'].iterrows()}
        self.calcfind_dict = {row['Code']: row['Meaning'] for _, row in df_legends[df_legends['Header in export'] == 'calcfind'].iterrows()}
        self.calcdistri_dict = {row['Code']: row['Meaning'] for _, row in df_legends[df_legends['Header in export'] == 'calcdistri'].iterrows()}
        
        # Only use 2D mammograms for now
        self.df = self.df[self.df[EMBED_IMAGE_TYPE_COL].isin(['2D'])]
        self.df[EMBED_PATH_COL] = self.df[EMBED_PATH_COL].apply(EMBED_PATH_TRANS_FUNC)
        
        if self.structural_cap or self.natural_cap:
            self.max_words = 144
        
        sub_train_set = split == 'train' or train_sub_set
        if data_pct != 1.0 and sub_train_set:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)
        
        if self.pred_density:
            if split == 'train':
                density_file = EMBED_TRAIN_PATH2DENSITY
            elif split == 'valid':
                density_file = EMBED_VALID_PATH2DENSITY
            elif split == 'test':
                density_file = EMBED_TEST_PATH2DENSITY
            else:
                raise ValueError(f"split {split} not supported")
            assert os.path.exists(density_file)
            self.path2density = pickle.load(open(density_file, "rb"))

        if self.balanced_test:
            if self.pred_density:
                assert os.path.exists(EMBED_BALANCED_DEN_TEST_PATH)
                print('### Using balanced test set with 4x500 examples...')
                # Note this also contains the density label
                self.balanced_test_path = pickle.load(open(EMBED_BALANCED_DEN_TEST_PATH, "rb"))
            else:
                assert os.path.exists(EMBED_BALANCED_TEST_PATH)
                print('### Using balanced test set with 7x200 examples...')
                self.balanced_test_path = pickle.load(open(EMBED_BALANCED_TEST_PATH, "rb"))
        else:
            self.balanced_test_path = None
        if self.small_balanced_train:
            if self.pred_density:
                assert os.path.exists(EMBED_BALANCED_DEN_TRAIN_PATH)
                print('### Using balanced train set with 4x550 examples...')
                # Note this also contains the density label
                self.balanced_train_path = pickle.load(open(EMBED_BALANCED_DEN_TRAIN_PATH, "rb"))
            else:
                assert os.path.exists(EMBED_BALANCED_TRAIN_PATH)
                print('### Using balanced train set with 7x550 examples...')
                self.balanced_train_path = pickle.load(open(EMBED_BALANCED_TRAIN_PATH, "rb"))
        else:
            self.balanced_train_path = None
        
        self.filenames, self.path2sent, self.path2label = self.load_text_data(split)
        
        self.tokenizer = get_tokenizer(llm_type)

        self.orig_crop_transform = DataTransforms(True, 256, 224)


    def load_text_data(self, split):
        base_filename = f"{split}_captions.pickle"
        if self.llm_type != 'gpt':
            base_filename = base_filename.replace(".pickle", f"_{self.llm_type}.pickle")
        if self.structural_cap:
            base_filename = base_filename.replace(".pickle", "_structural.pickle")
        elif self.simple_cap:
            base_filename = base_filename.replace(".pickle", "_simple.pickle")
        elif self.natural_cap:
            base_filename = base_filename.replace(".pickle", "_natural.pickle")
        filepath = os.path.join(EMBED_DATA_DIR, base_filename)
        
        if not os.path.isfile(filepath):
            print(f"### Caption file {filepath} does not exist. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            st = time.time()
            print(f"### Loading captions from {filepath}...")
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)
            print(f"### Loaded captions in {time.time() - st:.2} seconds")
        
        # Some of the paths in the dataframe are not in the captions
        filenames = []
        path2label = {}

        print("### extract label from captions...")
        for p, sentences in tqdm(path2sent.items()):
            # Only use the test image from balanced test set during test time
            if self.split == 'test' and self.balanced_test and p not in self.balanced_test_path.keys():
                continue
            # Only use the train image from balanced test set during training time
            if self.split == 'train' and self.small_balanced_train and p not in self.balanced_train_path.keys():
                continue
            # Extract BI-RAS label from the last sentence
            if self.pred_density:
                if p not in self.path2density.keys():
                    print(f"### {p} not in density map")
                    continue
                # Ignore male images
                label = self.path2density[p] - 1
                if label == 4:
                    continue
                path2label[p] = label
                filenames.append(p)
            else:
                if self.structural_cap:
                    sent = sentences[-2]
                    sent = sent[sent.find('bi rads'):]
                elif self.natural_cap:
                    sent = sentences[-1]
                    sent = sent[sent.find('bi rads'):]
                elif self.simple_cap:
                    sent = sentences[-1]
                    sent = sent[sent.find('birads'):]
                else:
                    sent = sentences[0]
                    sent = sent[sent.find('birads'):]
                for w in sent.split(' '):
                    if w.isdigit():
                        assert int(w) in set(EMBED_LETTER_TO_BIRADS.values())
                        path2label[p] = int(w)
                        filenames.append(p)
                        break
        print(np.unique(list(path2label.values()), return_counts=True))
        return filenames, path2sent, path2label
    
    def _create_captions_(self, row, meta_only=False):
        target_side = row[EMBED_SIDE_COL]
        anno_row = self.df_anno[self.df_anno[EMBED_SID_COL] == row[EMBED_SID_COL]]
        anno_full_row = self.df_anno_full[self.df_anno_full[EMBED_SID_COL] == row[EMBED_SID_COL]]
        if target_side in anno_row[EMBED_FINDING_SIDE_COL].tolist():
            anno_row = anno_row[anno_row[EMBED_FINDING_SIDE_COL] == target_side].iloc[0]
            anno_full_row = anno_full_row[anno_full_row[EMBED_FINDING_SIDE_COL] == target_side].iloc[0]
        else:
            anno_row = anno_row.iloc[0]
            anno_full_row = anno_full_row.iloc[0]
        # use the first annotation
        
        label_cnt = 0
        # if critical information is missing
        missing_info = False
        
        if self.structural_cap:
            captions = ""
            procedure = row[EMBED_PROCEDURE_COL]
            if check_element_type(procedure):
                captions += EMBED_PROCEDURE + procedure
                captions += "; "
                label_cnt += 1
            else:
                missing_info = True

            reason = EMBED_PROCEDURE2REASON_FUNC(procedure)
            if check_element_type(reason):
                captions += EMBED_REASON + reason
                captions += "; "
                label_cnt += 1
            else:
                missing_info = True

            age = anno_row[EMBED_AGE_COL]
            race = anno_row[EMBED_RACE_COL]
            ethnic = anno_row[EMBED_ETHNIC_COL]
            ethnic = "Non-Hispanic or Latino" if ethnic != "Hispanic or Latino" else ethnic
            patient_cap = EMBED_PATIENT_INFO_CAPTION
            patient_cap = patient_cap.replace("{{RACE}}", race if check_element_type(race) else "unknown")
            patient_cap = patient_cap.replace("{{ETHNIC}}", ethnic if check_element_type(ethnic) else "unknown")
            patient_cap = patient_cap.replace("{{AGE}}", str(int(age)) if check_element_type(age) else "unknown")
            captions += EMBED_PATIENT + patient_cap + " "
            label_cnt += 1

            image_type = row[EMBED_IMAGE_TYPE_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            image_cap = EMBED_IMAGE_INFO_CAPTION
            image_cap = image_cap.replace("{{IMAGE_TYPE}}", image_type if check_element_type(image_type) else "unknown")
            image_cap = image_cap.replace("{{SIDE}}", EMBED_SIDES_DESC[side] if check_element_type(side, EMBED_SIDES_DESC.keys()) else "unknown")
            image_cap = image_cap.replace("{{VIEW}}", view if check_element_type(view) else "unknown")
            captions += EMBED_IMAGE + image_cap + " "
            label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info
            
            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += EMBED_DENSITY + EMBED_BREAST_COMPOSITION_CAPTION.replace("{{DENSITY}}",density_desc)
                if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                    captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            calc_find = False
            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                mass_info = EMBED_MASS_CAPTION[asses]
                shape_code = anno_full_row[EMBED_MASS_SHAPE_COL]
                density_code = anno_full_row[EMBED_MASS_DENSITY_COL]
                if check_element_type(shape_code, self.massshape_dict.keys()) and check_element_type(density_code, self.massdensity_dict.keys()):
                    mass_info += EMBED_MASS_EXTRA_CAPTION.replace("{{SHAPE}}", self.massshape_dict[shape_code]).replace("{{DENSITY}}", self.massdensity_dict[density_code])
                captions += EMBED_FINDINGS + EMBED_FINDS_CAPTION + mass_info + " "
                
                calc_find_code = anno_full_row[EMBED_CALC_FIND_COL]
                calc_distri_code = anno_full_row[EMBED_CALC_DIST_COL]
                if check_element_type(calc_find_code, self.calcfind_dict.keys()) and check_element_type(calc_distri_code, self.calcdistri_dict.keys()):
                    calc_info = EMBED_CALC_FINDS_CAPTION.replace("{{SHAPE}}", self.calcfind_dict[calc_find_code]).replace("{{DISTRI}}", self.calcdistri_dict[calc_distri_code])
                    captions += calc_info + " "
                    calc_find = True
                
                birads = EMBED_LETTER_TO_BIRADS[asses]
                impression_desc = EMBED_BIRADS_DESC[asses]
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace("{{BIRADS}}", str(birads)).replace("{{BIRADS_DESC}}", impression_desc)
                
                captions += EMBED_ASSESSMENT + EMBED_ASSESSMENT_CAPTION[asses]
                label_cnt += 1
                
                assert '{{' not in captions
                # dev
                # if calc_find:
                #     print(captions)
            else:
                missing_info = True
        elif self.natural_cap:
            captions = EMBED_NATURE_BASE_CAPTION

            procedure = row[EMBED_PROCEDURE_COL]
            reason = EMBED_PROCEDURE2REASON_FUNC(procedure)
            if check_element_type(reason):
                captions = captions.replace("{{REASON}}", reason)
            else:
                captions = captions.replace("{{REASON}}", '')

            age = anno_row[EMBED_AGE_COL]
            race = anno_row[EMBED_RACE_COL]
            ethnic = anno_row[EMBED_ETHNIC_COL]
            ethnic = "Non-Hispanic or Latino" if ethnic != "Hispanic or Latino" else ethnic
            patient_cap = EMBED_PATIENT_INFO_CAPTION
            patient_cap = patient_cap.replace("{{RACE}}", race if check_element_type(race) else "unknown")
            patient_cap = patient_cap.replace("{{ETHNIC}}", ethnic if check_element_type(ethnic) else "unknown")
            patient_cap = patient_cap.replace("{{AGE}}", str(int(age)) if check_element_type(age) else "unknown")
            captions += patient_cap + " "
            label_cnt += 1

            image_type = row[EMBED_IMAGE_TYPE_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            image_cap = EMBED_NATURE_IMAGE_CAPTION
            image_cap = image_cap.replace("{{SIDE}}", EMBED_SIDES_DESC[side] if check_element_type(side, EMBED_SIDES_DESC.keys()) else "unknown")
            image_cap = image_cap.replace("{{VIEW}}", view if check_element_type(view) else "unknown")
            captions += image_cap + " "
            label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info
            
            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += EMBED_BREAST_COMPOSITION_CAPTION.replace("{{DENSITY}}",density_desc)
                if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                    captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                mass_info = EMBED_MASS_CAPTION[asses]
                shape_code = anno_full_row[EMBED_MASS_SHAPE_COL]
                density_code = anno_full_row[EMBED_MASS_DENSITY_COL]
                if check_element_type(shape_code, self.massshape_dict.keys()) and check_element_type(density_code, self.massdensity_dict.keys()):
                    mass_info += EMBED_MASS_EXTRA_CAPTION.replace("{{SHAPE}}", self.massshape_dict[shape_code]).replace("{{DENSITY}}", self.massdensity_dict[density_code])
                captions += EMBED_FINDS_CAPTION + mass_info + " "
                
                calc_find_code = anno_full_row[EMBED_CALC_FIND_COL]
                calc_distri_code = anno_full_row[EMBED_CALC_DIST_COL]
                if check_element_type(calc_find_code, self.calcfind_dict.keys()) and check_element_type(calc_distri_code, self.calcdistri_dict.keys()):
                    calc_info = EMBED_CALC_FINDS_CAPTION.replace("{{SHAPE}}", self.calcfind_dict[calc_find_code]).replace("{{DISTRI}}", self.calcdistri_dict[calc_distri_code])
                    captions += calc_info + " "
                
                birads = EMBED_LETTER_TO_BIRADS[asses]
                impression_desc = EMBED_BIRADS_DESC[asses]
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace("{{BIRADS}}", str(birads)).replace("{{BIRADS_DESC}}", impression_desc)
                
                assert '{{' not in captions
            else:
                missing_info = True
        else:
            # Start with base caption
            captions = BREAST_BASE_CAPTION
            
            if not self.simple_cap:
                # provide extra side, view, density information
                side = row[EMBED_SIDE_COL]
                if check_element_type(side, EMBED_SIDES_DESC.keys()):
                    captions += BREAST_SIDE_CAPTION + EMBED_SIDES_DESC[side]
                    captions += " "
                    label_cnt += 1
                
                view = row[EMBED_VIEW_COL]
                if check_element_type(view):
                    captions += BREAST_VIEW_CAPTION + view
                    captions += " "
                    label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info

            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += BREAST_DENSITY_CAPTION + str(int(density)) + ":" + density_desc + "."
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                asses_desc = EMBED_BIRADS_DESC[asses]
                birads = EMBED_LETTER_TO_BIRADS[asses]
                captions += BREAST_BIRADS_CAPTION + str(birads) + ":" + asses_desc + "."
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

        return captions, label_cnt, missing_info

    def create_path_2_sent_mapping(self):
        sent_lens = []
        path2sent = {}
        for i, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # Find annotations for this image
            # Can be more than 1 annotations
            captions, label_cnt, missing_info = self._create_captions_(row)

            # Skip the image if there is no label
            if label_cnt == 0 or missing_info:
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
                path2sent[row[EMBED_PATH_COL]] = study_sent
        
        sent_lens = np.array(sent_lens)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}] {len(sent_lens)}"
        )
        
        return path2sent
    
    def __len__(self):
        return len(self.filenames)
    
    def random_mask(self, tokens, mask_ratio=0.5):
        # Unused
        return tokens
    
    def get_caption(self, path, series_sents=None):
        if series_sents is None:
            series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])
        masked_ids = self.random_mask(tokens['input_ids'])
        tokens['masked_ids'] = masked_ids

        return tokens, x_len
    
    def get_birads_one_hot_label(self, index, get_full=False):
        multi_hot_label = torch.zeros(len(EMBED_LETTER_TO_BIRADS))
        key = self.filenames[index]
        asses = self.path2label[key]
        multi_hot_label[asses] = 1
        return multi_hot_label
    
    def get_density_one_hot_label(self, index, get_full=False):
        multi_hot_label = torch.zeros(len(EMBED_DENSITY_DESC) - 1)
        key = self.filenames[index]
        density = self.path2label[key]
        multi_hot_label[density] = 1
        return multi_hot_label
    
    def __cls_getitem__(self, index):
        key = self.filenames[index]
        if self.pred_density:
            one_hot_label = self.get_density_one_hot_label(index)
        else:
            one_hot_label = self.get_birads_one_hot_label(index)
        if self.zero_shot_caps is None or self.instance_test_cap:
            zero_shot_caps = []
            zero_shot_caps_len = []
            # get base caption
            if self.instance_test_cap:
                target_row = self.df[self.df[EMBED_PATH_COL] == key].iloc[0]
                base_captions = self._create_captions_(target_row, meta_only=True)[0]
            else:
                base_captions = ''
            # get zero-shot captions based on classes
            if self.pred_density:
                for density, density_desc in EMBED_DENSITY_DESC.items():
                    if density == 5:
                        continue
                    captions = base_captions + BREAST_BASE_CAPTION + BREAST_DENSITY_CAPTION + str(density) + ": " + density_desc + "."
                    captions = captions.replace("\n", " ").lower()
                    cap, cap_len = self.get_caption(None, [captions])
                    zero_shot_caps.append(cap)
                    zero_shot_caps_len.append(cap_len)
            else:
                for asses, birads_desc in EMBED_BIRADS_DESC.items():
                    birads = EMBED_LETTER_TO_BIRADS[asses]
                    captions = base_captions + BREAST_BASE_CAPTION + BREAST_BIRADS_CAPTION + str(birads) + ": " + birads_desc + "."
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
        imgs, orig_img = read_from_dicom(key, self.imsize, self.transform, True)
        orig_img = self.orig_crop_transform(orig_img)
        return imgs, self.zero_shot_caps, self.zero_shot_caps_len, key, one_hot_label, orig_img
    
    def __getitem__(self, index):
        if self.cls_prompt:
            return self.__cls_getitem__(index)
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs, orig_img = read_from_dicom(key, self.imsize, self.transform, True)
        
        if self.pred_density:
            one_hot_label = self.get_density_one_hot_label(index)
        else:
            one_hot_label = self.get_birads_one_hot_label(index)
        orig_img = self.orig_crop_transform(orig_img)
        # No need to unpaired text
        return imgs, caps, cap_len, key, one_hot_label, orig_img


