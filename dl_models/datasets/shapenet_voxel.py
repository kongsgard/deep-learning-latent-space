import numpy as np
import os
import torch
from torch.utils import data

from utils.voxel_utils import get_voxels_from_mat


class ShapeNetDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, config):
        """Set the path for Data."""
        self.data_folder = config.data_folder
        self.listdir = os.listdir(self.data_folder)
        self.cube_len = config.cube_len

    def __getitem__(self, index):
        with open(self.data_folder + self.listdir[index], "rb") as f:
            volume = np.asarray(get_voxels_from_mat(f, self.cube_len), dtype=np.float32)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)


class ShapeNetDataLoader:
    def __init__(self, config):
        self.config = config

        if config.data_mode == "shapes":
            dataset = ShapeNetDataset(self.config)

            self.dataset_len = dataset.__len__()

            self.num_iterations = (self.dataset_len + self.config.batch_size - 1) // self.config.batch_size

            self.loader = data.DataLoader(dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          num_workers=config.data_loader_workers,
                                          pin_memory=config.pin_memory)
        else:
            raise Exception("Please specify in the config json a specified mode in data_mode")

    def finalize(self):
        pass
