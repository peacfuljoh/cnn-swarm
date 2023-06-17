
import os
import time

import torch


TS_INIT = str(int(time.time() * 1e6))


# local dirs
HOME_DIR = os.environ['HOME_DIR']
TORCH_MODEL_DIR = os.path.join(HOME_DIR, 'torch_models')
TORCH_DATA_ROOT = os.path.join(HOME_DIR, 'torch_data')
TORCH_DATA_TEMP = os.path.join(TORCH_DATA_ROOT, 'temp')
TORCH_RUN_SUMMARY_DIR = os.path.join(HOME_DIR, 'torch_runs')
STATUS_DIR = os.path.join(HOME_DIR, 'status')

for dir_ in [TORCH_MODEL_DIR, TORCH_DATA_ROOT, TORCH_DATA_TEMP, TORCH_RUN_SUMMARY_DIR, STATUS_DIR]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)

# get cpu or gpu device for training
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")

# CIFAR10 info
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# status file
STATUS_FPATH = os.path.join(STATUS_DIR, TS_INIT + '.json')

# model info
MODEL_TYPES = ['NNConvResNetRGB']

# dataset info
DATASET_NAMES = ['CIFAR10']

# misc
MODEL_FILE_EXT = '.pth'

# model info file
MODEL_INFO_FPATH = os.path.join(TORCH_MODEL_DIR, 'model_info.json')

# data subsets
DATASET_SUBSETS = ['train', 'test']
