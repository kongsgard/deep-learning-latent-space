#!/usr/bin/env bash

# Use this format to run a network with the given config
#python3 main.py PATH_OF_THE_CONFIG_FILE

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

#python main.py configs/dcgan_exp_0.json
python main.py configs/3dgan_exp_0.json