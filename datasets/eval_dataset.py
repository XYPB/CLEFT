import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from .constants_val import *
from .utils import get_imgs, read_from_dicom, get_tokenizer
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class RSNADataset(data.Dataset):
    def __init__(self, split='train', transform=None, data_pct=1.0,
                 imsize=256, max_words=64, sent_num=3, keep_size=False,
                 llm_type="gpt", prompt_ft=False, *args, **kwargs):
        super().__init__()
        if not os.path.exists(RSNA_DATA_DIR):
            raise RuntimeError(f"{RSNA_DATA_DIR} does not exist!")
        
        self.llm_type = llm_type
        self.transform = transform
        self.imsize = imsize
        self.split = split
        self.prompt_ft = prompt_ft
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        self.zero_shot_one_hot_label = None
        
        if split == 'train' or split == 'valid':
            self.data_path = RSNA_TRAIN_DATA_PATH
            data_csv = RSNA_TRAIN_CSV
        elif split == 'test':
            self.data_path = RSNA_TRAIN_DATA_PATH
            data_csv = RSNA_TEST_CSV
        self.df = pd.read_csv(data_csv)
        self.filenames, self.path2label = self.load_text_data()
        
        if data_pct != 1.0 and split == "train":
            random.seed(42)
            self.filenames = random.sample(self.filenames, int(data_pct * len(self.filenames)))
            if keep_size:
                self.filenames = self.filenames * int(1/data_pct)
        
        self.tokenizer = get_tokenizer(llm_type)
        self.max_words = max_words

        
    def load_text_data(self):
        filenames = []
        path2label = {}
        for idx, row in self.df.iterrows():
            filepath = os.path.join(self.data_path, row['patientId'] + '.dcm')
            label = row['Target']
            filenames.append(filepath)
            path2label[filepath] = label
        print(np.unique(list(path2label.values()), return_counts=True))
        return filenames, path2label
    
    def __len__(self):
        return len(self.filenames)
    
    def get_caption(self, path, series_label=None):
        if series_label is None:
            series_label = [self.path2label[path]]
            
        series_sents = []
        for label in series_label:
            prompt = '' if self.prompt_ft else CHEXPERT_BASE_CAPTION
            if label == 0:
                prompt += "no findings"
            else:
                prompt += "pneumothorax"
            series_sents.append(prompt.replace("\n", " ").lower())

        tokens = self.tokenizer(
            series_sents,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        tokens['masked_ids'] = tokens['input_ids'].clone()

        return tokens, x_len
    
    def __getitem__(self, index):
        path = self.filenames[index]
        img = read_from_dicom(path, self.imsize, self.transform)
        label = self.path2label[path]
        one_hot_labels = torch.zeros(2)
        one_hot_labels[label] = 1 
        
        if self.zero_shot_caps is None:
            stacked_caps, zero_shot_caps_len = self.get_caption(None, [0, 1])
            self.zero_shot_caps = stacked_caps
            self.zero_shot_caps_len = torch.tensor(zero_shot_caps_len)
        return img, self.zero_shot_caps, self.zero_shot_caps_len, path, one_hot_labels, img
    
