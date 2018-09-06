from __future__ import print_function, division

import numpy as np
import torchvision
import matplotlib.pyplot as plt

import argparse
from utils.config import *

from datasets.celebA import CelebADataLoader

# plt.ion()  # interactive mode


def imshow(images, title=None):
    """Show tensor images"""
    images = images / 2 + 0.5  # Unnormalize
    images = images.numpy().transpose((1, 2, 0))
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause for a moment to update plots


def main():
    # Parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()
    config, _ = get_config_from_json(args.config)

    # Load the data
    dataloader = CelebADataLoader(config)

    # Get a batch of training data
    inputs, classes = next(iter(dataloader.loader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title='CelebA')
    input("Press Enter to continue...")


if __name__ == '__main__':
    main()


