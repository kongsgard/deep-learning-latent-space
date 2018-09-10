import torch
import torchvision
import visdom

from agents.base import BaseAgent
from datasets.shapenet import ShapeNetDataLoader
from utils.voxel_utils import plot_voxels_in_visdom


class ThreeDimensionalGANAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # Define dataloader
        self.dataloader = ShapeNetDataLoader(self.config)

        # Plot first input
        inputs = next(iter(self.dataloader.loader))
        vis = visdom.Visdom()
        plot_voxels_in_visdom(torch.Tensor.numpy(inputs[0]), vis)

    def load_checkpoint(self, file_name):
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        pass

    def run(self):
        pass

    def train(self):
        pass

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def finalize(self):
        pass
