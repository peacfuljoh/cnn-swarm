from typing import List, Optional

import torch
from torch import nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from ossr_utils.io_utils import load_json, save_json

from src.docker_exps.ml.nn_models import NNConvResNetRGB
from src.docker_exps.constants import TORCH_DATA_ROOT, MODEL_INFO_FPATH


def get_cifar10_data(batch_size: int = 64,
                     train: bool = True,
                     test: bool = True,
                     shuffle: bool = True,
                     example_idxs_train: Optional[List[int]] = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if train:
        trainset = torchvision.datasets.CIFAR10(root=TORCH_DATA_ROOT, train=True, download=True, transform=transform) # 50k
        if example_idxs_train is not None:
            trainset = torch.utils.data.Subset(trainset, range(example_idxs_train[0], example_idxs_train[1]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    else:
        trainset, trainloader = None, None
    if test:
        testset = torchvision.datasets.CIFAR10(root=TORCH_DATA_ROOT, train=False, download=True, transform=transform) # 10k
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    else:
        testset, testloader = None, None

    data_func = lambda x: x / 2 + 0.5 # reverse of normalize operation

    return trainset, testset, trainloader, testloader, data_func

def init_net(lr,
             momentum,
             scheduler_mode: Optional[str] = None,
             scheduler_params: Optional[dict] = None,
             num_epochs: Optional[int] = None):
    # net = NNConvLeNetRGB()
    net = NNConvResNetRGB()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    scheduler = None
    if scheduler_mode == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_params['gamma'])
    if scheduler_mode == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2,
                                                         threshold=0.2, min_lr=lr * 0.01)
    if scheduler_mode == 'multistep':
        num_change_epochs = num_epochs // 3
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [num_change_epochs, num_change_epochs * 2], gamma=0.1)
    return net, criterion, optimizer, scheduler

def update_model_info_file(mode: str,
                           model_info: dict):
    if mode == 'insert':
        model_id = model_info['model_id']
        model_info_no_id = {key: val for key, val in model_info.items() if key != 'model_id'}
        print('Performing an insert to the model info file\n  model_id: {}\n  content: {}'.format(model_id, model_info_no_id))
        d = load_json(MODEL_INFO_FPATH)
        d[model_id] = model_info_no_id
        save_json(MODEL_INFO_FPATH, d)

