import numpy as np
import os
import torch
from torch.utils import data

from tensorpack import dataflow

def resample_pcd(pcd, n):
        """Drop or duplicate points so that pcd has exactly n points"""
        index = np.random.permutation(pcd.shape[0])
        if index.shape[0] < n:
            index = np.concatenate([index, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
        return pcd[idx[:n]]

class ShapeNetPointCloudDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, config, dataset_mode):
        """Set the path for Data."""
        self.data_folder = config.data_folder
        self.dataset_mode = dataset_mode

        self.df = dataflow.LMDBSerializer.load(self.data_folder + self.dataset_mode + '.lmdb', shuffle=False)


    def __getitem__(self, index):
        id, input, gt = next(self.df.get_data())
        return id, torch.FloatTensor(input), torch.FloatTensor(gt)
        
    def __len__(self):
        return self.df.size()


class ShapeNetPointCloudDataLoader:
    def __init__(self, config, dataset_mode):
        self.config = config
        self.dataset_mode = dataset_mode

        if config.data_mode == "shapes":
            dataset = ShapeNetPointCloudDataset(self.config, self.dataset_mode)

            self.dataset_len = dataset.__len__()

            self.num_iterations = (self.dataset_len + self.config.batch_size - 1) // self.config.batch_size

            self.loader = data.DataLoader(dataset,
                                          batch_size=self.config.batch_size,
                                          shuffle=True,
                                          num_workers=config.data_loader_workers,
                                          pin_memory=config.pin_memory)
        else:
            raise Exception("Please specify in the config json a specified mode in data_mode")

    def finalize(self):
        pass