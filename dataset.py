import numpy as np
from pathlib import Path
import os
import pdb
from easydict import EasyDict
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import pytorch_lightning as pl
import time
import albumentations as A



class SiameseDatset(Dataset):
    def __init__(
        self,
        data_path,
        mode, # train, val or test
        feat_dim,
    ):
        super(SiameseDatset, self).__init__()
        self.pkl_fp = data_path
        self.mode = mode
        self.feat_dim = feat_dim
        self._init_paths_and_labels()
    

    def _init_paths_and_labels(self):
        with open(self.pkl_fp, 'rb') as f:
            self.data = pickle.load(f)
            

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        item = self.data[index]
        
        v1 = item['rs_emb']
        v2 = item['jd_emb']
        is_match = item['is_match']
            
        v1 = torch.tensor(v1).to(torch.float)
        v2 = torch.tensor(v2).to(torch.float)
        is_match = torch.tensor(is_match).to(torch.float)
        
        return v1, v2, is_match



class SiameseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_pkl_fp,
        val_pkl_fp, 
        test_pkl_fp,
        predict_pkl_fp,
        data_cfg: dict, 
        training_cfg: dict
    ):
        super(SiameseDataModule, self).__init__()
        self.train_pkl_fp = train_pkl_fp
        self.val_pkl_fp = val_pkl_fp
        self.test_pkl_fp = test_pkl_fp
        self.predict_pkl_fp = predict_pkl_fp
        self.data_cfg = EasyDict(data_cfg)
        self.training_cfg = EasyDict(training_cfg)

    
    def setup(self, stage):
        if stage == 'fit' or stage == 'validate':
            self.train_ds = SiameseDatset(data_path=self.train_pkl_fp, mode='train', **self.data_cfg)
            self.val_ds = SiameseDatset(data_path=self.val_pkl_fp, mode='val', **self.data_cfg)
        elif stage == 'test':
            self.test_ds = SiameseDatset(data_path=self.test_pkl_fp, mode='test', **self.data_cfg)
        elif stage == 'predict':
            self.predict_ds = SiameseDatset(data_path=self.predict_pkl_fp, mode='predict', **self.data_cfg)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=self.training_cfg.shuffle_train, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )


    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, 
            batch_size=self.training_cfg.bs, 
            shuffle=False, 
            num_workers=self.training_cfg.num_workers,
            pin_memory=False
        )



if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    config.data.training_cfg.num_workers = 0

    ds_module = SiameseDataModule(**config.data)
    ds_module.setup('validate')
    for i, item in enumerate(ds_module.val_dataloader()):
        v1, v2, labels = item
        print(v1.shape, v2.shape)
        print(labels)
        pdb.set_trace()