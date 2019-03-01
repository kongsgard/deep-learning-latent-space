# deep-learning-latent-space
Learning Latent Space of Compliant Objects with Deep Learning for Robotic Grasping

## Installation
Make sure you have CUDA before proceeding with the installation.
```shell
conda env create -f torch-env.yml
conda activate torch
```

#### Build chamfer distance 
```shell
conda activate torch
cd dl_models/utils/pcd/chamfer
python setup.py install
```

## Usage
- Navigate to the /dl_models folder: `cd dl_models`
- Activate the conda environment: `conda activate torch`
- Start visdom for visualization during training: `visdom`
- [Optional] See training summaries on tensorboardX: `tensorboard --logdir experiments/<name_of_experiment>`
- Train the network: `python main.py configs/pcn.json`