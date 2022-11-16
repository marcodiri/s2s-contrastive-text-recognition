import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torchvision
from kornia import tensor_to_image
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from dataset import BatchBalancedDataset, HierarchicalDataset


class WordsDataModule(pl.LightningDataModule):
    def __init__(self, opt, tranform=None):
        super().__init__()
        self.opt = opt
        self.transform = tranform

    def show_batch(self, win_size=(10, 10)):
        def _to_vis(data):
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))

        # set num_workers to 0 temporarily since using more workers 
        # for a single batch would slow down
        num_workers = self.opt.workers
        self.opt.workers = 0
        imgs, labels = next(iter(self.train_dataloader()))
        self.opt.workers = num_workers
        
        plt.figure(figsize=win_size)
        plt.title("original")
        plt.imshow(_to_vis(imgs))
        if self.transform:
            imgs_aug = self.transform(imgs)
            plt.figure(figsize=win_size)
            plt.title("augmented")
            plt.imshow(_to_vis(imgs_aug))
    
    def setup(self, stage:str = None):
        if stage == "fit" or stage is None:
            self.dataset_train = BatchBalancedDataset(self.opt)
            self.dataset_val = HierarchicalDataset(
                root=self.opt.valid_data, 
                opt=self.opt)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        imgs, labels = batch
        if self.transform:
            imgs = self.transform(imgs)
        return imgs, labels

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
        return DataLoader(
            self.dataset_train, 
            batch_sampler=BatchSampler(
                sampler,
                batch_size=self.opt.batch_size,
                drop_last=False),
            num_workers=int(self.opt.workers),
            pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, 
            batch_size=self.opt.batch_size,
            num_workers=int(self.opt.workers),
            pin_memory=True)
