
import os
import time

import torch



# local dirs
HOME_DIR = os.environ['HOME_DIR']
TORCH_MODEL_DIR = os.path.join(HOME_DIR, 'torch_models')
TORCH_DATA_ROOT = os.path.join(HOME_DIR, 'torch_data')
TORCH_DATA_TEMP = os.path.join(TORCH_DATA_ROOT, 'temp')
TORCH_RUN_SUMMARY_DIR = os.path.join(HOME_DIR, 'torch_runs')

# get cpu or gpu device for training
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {DEVICE} device")

# CIFAR10 info
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Flask
FLASK_HOST = os.environ['FLASK_HOST']
FLASK_PORT = int(os.environ['FLASK_PORT'])

# status file
STATUS_DIR = os.path.join(HOME_DIR, 'status')
if not os.path.exists(STATUS_DIR):
    os.mkdir(STATUS_DIR)
STATUS_FPATH = os.path.join(STATUS_DIR, str(int(time.time() * 1e6)) + '.json')
