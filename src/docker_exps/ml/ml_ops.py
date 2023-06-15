from typing import List, Optional

import torch
from torch import nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np

from src.docker_exps.ml.nn_models import NNConvResNetRGB
from src.docker_exps.ml.nn_utils import eval_classifier_test_acc
from src.docker_exps.constants import TORCH_DATA_ROOT



def augment_data(x):
    # left-right flip
    if np.random.rand() > 0.5:
        x = torch.flip(x, [3])

    # jitter (zero-pad and re-crop)
    if 0:
        im_dims = x.shape[2:]
        num_pad = 4
        jit = np.random.randint(0, (2 * num_pad), 2)
        x = torch.nn.functional.pad(x, (num_pad, num_pad, num_pad, num_pad), value=0) # zero-pad last two dims
        x = x[:, :, jit[0]:jit[0] + im_dims[0], jit[1]:jit[1] + im_dims[1]] # crop

    if 0:
        data_func = lambda x: x / 2 + 0.5
        vis = DLViz(data_func=data_func)
        vis.show_grid_from_tensor(x, 4, 4, show=True)

    return x

def train(trainloader,
          net,
          criterion,
          optimizer,
          num_epochs,
          board = None,
          testloader = None,
          scheduler_mode = None,
          scheduler = None,
          augment: bool = True) \
        -> List[float]:
    """Train"""
    epoch_loss = []
    num_train = len(trainloader) * trainloader.batch_size
    n_update_board = (num_train / trainloader.batch_size) // 10

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # start from 0'th (default) or desired item
            # get the inputs
            inputs, labels = data
            if augment:
                inputs = augment_data(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # labels = (labels.reshape(len(labels), 1) == torch.arange(outputs.shape[1])).type(torch.float)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            loss_iter = loss.item()
            running_loss += loss_iter

            if i % n_update_board == (n_update_board - 1):
                if board is not None:
                    num_exs = trainloader.batch_size * (i + 1)
                    num_batch_tot = epoch * num_train + num_exs
                    num_epochs = epoch + num_exs / num_train # TODO: switch to using this
                    board.add_scalar('training loss', running_loss / num_exs, num_batch_tot)
                    if testloader is not None:
                        acc, acc_top3 = eval_classifier_test_acc(testloader, net, top_n=3)
                        board.add_scalar('test accuracy', acc, num_batch_tot)
                        # board.add_scalar('test top-3 accuracy', acc_top3, num_batch_tot)

        epoch_loss.append(running_loss)
        print("epoch %d, loss = %.3f" % (epoch + 1, epoch_loss[epoch]))

        if scheduler_mode in ['plateau']:
            scheduler.step(running_loss)
        else:
            scheduler.step()

    print('Finished Training')

    return epoch_loss

def get_cifar10_data(batch_size: int = 64,
                     train: bool = True,
                     test: bool = True,
                     shuffle: bool = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if train:
        trainset = torchvision.datasets.CIFAR10(root=TORCH_DATA_ROOT, train=True, download=True, transform=transform) # 50k
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