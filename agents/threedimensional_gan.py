from tqdm import tqdm

import torch
import visdom

from agents.base import BaseAgent
from datasets.shapenet import ShapeNetDataLoader
from utils.misc import print_cuda_statistics
from utils.voxel_utils import plot_voxels_in_visdom


class ThreeDimensionalGANAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # Define dataloader
        self.dataloader = ShapeNetDataLoader(self.config)

        # Initialize counter
        self.current_epoch = 0
        self.current_iteration = 0

        # Set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device - enable it!")
        self.cuda = self.is_cuda & self.config.cuda

        # Set the manual seed for torch
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            torch.cuda.manual_seed_all(self.config.seed)
            self.logger.info("Program will run on ***GPU-CUDA***")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Program will run on ***CPU***")

    def load_checkpoint(self, file_name):
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        pass

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("CTRL+C was pressed - wait to finalize...")

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()

    def train_one_epoch(self):
        # Initialize tqdm batch
        tqdm_batch = tqdm(self.dataloader.loader, total=self.dataloader.num_iterations,
                          desc="epoch-{}-".format(self.current_epoch))

        vis = visdom.Visdom()  # TODO: Move
        for curr_it, x in enumerate(tqdm_batch):
            if curr_it == 0:  # Todo: Refactor
                plot_voxels_in_visdom(torch.Tensor.numpy(x[0]), vis)

    def validate(self):
        pass

    def finalize(self):
        pass
