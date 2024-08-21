import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler



class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, num_workers, crop_size=224,
                 masked_lm_ratio=0.15, llm_type="gpt", prompt_ft=False,
                 train_split='train', valid_split='valid', test_split='test',
                 five_cls=False, train_sub_set=False, structural_cap=False, simple_cap=False,
                 natural_cap=False, instance_test_cap=False,
                 balanced_test=False, balance_training=False, small_balanced_train=False,
                 pred_density=False, keep_size=False):
        super().__init__()

        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.masked_lm_ratio = masked_lm_ratio
        self.llm_type = llm_type
        self.prompt_ft = prompt_ft
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        self.five_cls = five_cls
        self.train_sub_set = train_sub_set
        self.structural_cap = structural_cap
        self.simple_cap = simple_cap
        self.natural_cap = natural_cap
        self.instance_test_cap = instance_test_cap

        self.balanced_test = balanced_test
        self.balance_training = balance_training
        self.small_balanced_train = small_balanced_train
        self.pred_density = pred_density
        self.keep_size = keep_size

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None
        
        dataset = self.dataset(
            split=self.train_split, transform=transform, data_pct=self.data_pct,
            masked_lm_ratio=self.masked_lm_ratio, llm_type=self.llm_type,
            prompt_ft=self.prompt_ft,
            five_cls=self.five_cls, simple_cap=self.simple_cap,
            train_sub_set=self.train_sub_set, structural_cap=self.structural_cap,
            natural_cap=self.natural_cap, instance_test_cap=self.instance_test_cap, 
            balanced_test=self.balanced_test, 
            small_balanced_train=self.small_balanced_train,
            pred_density=self.pred_density, keep_size=self.keep_size)
        
        if self.balance_training:
            num_samples = len(dataset)
            _, class_counts = np.unique(list(dataset.path2label.values()), return_counts=True)
            class_weights = 1. / class_counts
            weights = []
            for idx in range(num_samples):
                lb = dataset.path2label[dataset.filenames[idx]]
                weights.append(class_weights[lb])

            sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            ) 
        else:
            return DataLoader(
                dataset,
                pin_memory=True,
                drop_last=True,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split=self.valid_split, transform=transform, data_pct=self.data_pct,
            masked_lm_ratio=self.masked_lm_ratio, llm_type=self.llm_type,
            prompt_ft=self.prompt_ft,
            five_cls=self.five_cls, simple_cap=self.simple_cap,
            train_sub_set=self.train_sub_set, structural_cap=self.structural_cap,
            natural_cap=self.natural_cap, instance_test_cap=self.instance_test_cap,
            balanced_test=self.balanced_test,
            small_balanced_train=self.small_balanced_train,
            pred_density=self.pred_density, keep_size=self.keep_size)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split=self.test_split, transform=transform, data_pct=self.data_pct,
            masked_lm_ratio=self.masked_lm_ratio, llm_type=self.llm_type,
            prompt_ft=self.prompt_ft,
            five_cls=self.five_cls, simple_cap=self.simple_cap,
            train_sub_set=self.train_sub_set, structural_cap=self.structural_cap,
            natural_cap=self.natural_cap, instance_test_cap=self.instance_test_cap,
            balanced_test=self.balanced_test,
            small_balanced_train=self.small_balanced_train,
            pred_density=self.pred_density, keep_size=self.keep_size)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

# if __name__=="__main__":
