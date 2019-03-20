from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class ModelNet40Cls(data.Dataset):
    def __init__(self, config, mode, transforms=None, train=True, download=True):
        super().__init__()

        self.transforms = transforms
        self.data_base_directory = config.data_base_directory
        self.data_folder = config.data_folder
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(self.data_folder):
            zipfile = os.path.join(self.data_base_directory, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, self.data_base_directory))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.mode, self.num_points = mode, config.num_points
        if self.mode == "train":
            self.files = _get_data_files(os.path.join(self.data_folder, "train_files.txt"))
        else:
            self.files = _get_data_files(os.path.join(self.data_folder, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(self.data_base_directory, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

        self.randomize()

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.actual_number_of_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = pts
        self.actual_number_of_points = pts

    def randomize(self):
        self.actual_number_of_points = min(
            max(np.random.randint(self.num_points * 0.8, self.num_points * 1.2), 1),
            self.points.shape[1],
        )


class ModelNet40PointCloudDataloader:
    def __init__(self, config, dataset_mode, transforms):
        self.config = config
        self.dataset_mode = dataset_mode

        dataset = ModelNet40Cls(config=self.config, mode=self.dataset_mode, transforms=transforms)

        self.dataset_len = dataset.__len__()

        self.num_iterations = (self.dataset_len + self.config.batch_size - 1) // self.config.batch_size

        self.loader = data.DataLoader(dataset,
                                      batch_size=self.config.batch_size,
                                      shuffle=True,
                                      num_workers=config.data_loader_workers,
                                      pin_memory=config.pin_memory)

    def finalize(self):
        pass