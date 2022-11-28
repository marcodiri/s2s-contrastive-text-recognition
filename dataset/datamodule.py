from math import ceil
from os import path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from kornia import tensor_to_image
from torch.utils.data import (BatchSampler, DataLoader, WeightedRandomSampler,
                              default_collate)

from dataset.converter import (AttnLabelConverter, CTCLabelConverter,
                               CTCLabelConverterForBaiduWarpctc)
from dataset.dataset import BatchBalancedDataset, HierarchicalDataset


class WordsDataModule(pl.LightningDataModule):
    def __init__(self, opt, pretraining=False, tranform=None):
        super().__init__()
        self.opt = opt
        self.pretraining = pretraining
        self.transform = tranform
        
        if not self.pretraining:
            if 'CTC' in opt.Prediction:
                if opt.baiduCTC:
                    self.label_converter = CTCLabelConverterForBaiduWarpctc(opt.character)
                else:
                    self.label_converter = CTCLabelConverter(opt.character)
            else:
                self.label_converter = AttnLabelConverter(opt.character)
            opt.num_class = len(self.label_converter.character)

    def show_batch(self, imgs, win_size=(10, 10)):
        def _to_vis(data):
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=5))

        plt.figure(figsize=win_size)
        plt.title("original")
        plt.imshow(_to_vis(imgs))
    
    def setup(self, stage:str = None):
        log = open(path.join(self.opt.save_dir,
                             self.opt.exp_name, 'log_dataset.txt'),
                   'a')
        
        if stage == "fit" or stage is None:
            self.dataset_train = BatchBalancedDataset(self.opt)
            self.train_batches = ceil(
                len(self.dataset_train)/self.opt.batch_size)
            train_batches_log = f'Training batches: {self.train_batches}\n'\
                + '-' * 80 + '\n' + '-' * 80 + '\n'
            self.dataset_train.log += train_batches_log
            print(self.dataset_train.log)
            log.write(self.dataset_train.log)
            
            if not self.pretraining:
                self.dataset_val = HierarchicalDataset(
                    root=self.opt.valid_data, 
                    opt=self.opt)
                self.val_batches = ceil(
                    len(self.dataset_val)/self.opt.batch_size)
                val_batches_log = f'Validation batches: {self.val_batches}\n'\
                    + '-' * 80 + '\n'
                self.dataset_val.log += val_batches_log
                print(self.dataset_val.log)
                log.write(self.dataset_val.log)
            
        log.close()
        
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.pretraining:
            imgs, _ = batch
            imgs_cat = torch.cat((imgs, imgs))
            if self.transform:
                # self.show_batch(imgs)
                imgs_cat = self.transform(imgs_cat)
                # self.show_batch(imgs_cat)
            return imgs_cat
        else:
            imgs, labels, lengths = batch
            if self.transform:
                imgs = self.transform(imgs)
            return (imgs, labels, lengths)

    def train_dataloader(self):
        # calculate weight of each dataset depending on its length and opt.batch_ratio
        total_len = sum(self.dataset_train.dataset_len_list)
        if len(self.dataset_train.dataset_len_list) == 1:
            weights = np.ones(total_len)
        else:
            weights = [(1-_len/total_len) * float(self.opt.batch_ratio[i])
                    for i,_len in enumerate(self.dataset_train.dataset_len_list)]
            # expand weights to match dataset length
            weights = np.repeat(weights, self.dataset_train.dataset_len_list)
        sampler = WeightedRandomSampler(weights,total_len,replacement=False)
        if self.pretraining:
            return DataLoader(
                self.dataset_train, 
                batch_sampler=BatchSampler(
                    sampler,
                    batch_size=self.opt.batch_size,
                    drop_last=False),
                num_workers=int(self.opt.workers),
                pin_memory=True)
        else:
            return DataLoader(
                self.dataset_train, 
                batch_sampler=BatchSampler(
                    sampler,
                    batch_size=self.opt.batch_size,
                    drop_last=False),
                collate_fn=self.convert_collate,
                num_workers=int(self.opt.workers),
                pin_memory=True)
        
    def val_dataloader(self):
        if not self.pretraining:
            return DataLoader(
                self.dataset_val,
                batch_size=self.opt.batch_size,
                shuffle=False,
                collate_fn=self.convert_collate,
                num_workers=int(self.opt.workers),
                pin_memory=True)
        else:
            super().val_dataloader()

    def convert_collate(self, data):
        """Use self.converter to convert text labels into tensors"""
        imgs, labels = default_collate(data)
        tensor_labels, lengths = self.label_converter.encode(
            labels, 
            batch_max_length=self.opt.batch_max_length)
        return (imgs, tensor_labels, lengths)
