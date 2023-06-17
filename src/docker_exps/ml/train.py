"""Simple CNNs for grayscale and RGB images (CIFAR10, LeNet)"""

import os
from typing import List, Optional, Union
import time

import torch

from src.docker_exps.ml.nn_utils import eval_classifier_test_acc
from src.docker_exps.ml.ml_ops import train, get_cifar10_data, init_net, update_model_info_file
from src.docker_exps.constants import TORCH_MODEL_DIR, MODEL_FILE_EXT


BATCH_SIZE_CIFAR10 = 64


def train_on_cifar10(trainloader: torch.utils.data.Dataset,
                     testloader: torch.utils.data.Dataset,
                     model_id: str,
                     train_opts_spec: Optional[dict] = None):
    # visualize batch
    if 0:
        vis_cifar10_batch(nrows=8, ncols=8, show=True)

    # setup tensorboard
    # dts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # summary_dir = os.path.join(TORCH_RUN_SUMMARY_DIR, 'CIFAR10_' + dts)
    # board = Tensorboard(summary_dir)
    board = None

    if 0:
        vis = vis_cifar10_batch(nrows=8, ncols=8)
        board.add_image('CIFAR10_images', vis._im_grid)

    # training params
    train_opts = dict(
        num_epochs = 80,
        lr = 0.01,
        momentum = 0.5,
        scheduler_mode = 'multistep'
    )

    if train_opts_spec is not None:
        for key, val in train_opts_spec.items():
            train_opts[key] = val

    print('Training settings:')
    print(train_opts)

    num_epochs = train_opts['num_epochs']
    lr = train_opts['lr']
    momentum = train_opts['momentum']
    scheduler_mode = train_opts['scheduler_mode']

    # init, train, test
    if scheduler_mode == 'exponential':
        net, criterion, optimizer, scheduler = init_net(lr, momentum, scheduler_mode=scheduler_mode,
                                                        scheduler_params=dict(gamma=0.80))
    if scheduler_mode == 'plateau':
        net, criterion, optimizer, scheduler = init_net(lr, momentum, scheduler_mode=scheduler_mode)
    if scheduler_mode == 'multistep':
        net, criterion, optimizer, scheduler = init_net(lr, momentum, scheduler_mode=scheduler_mode,
                                                        num_epochs=num_epochs)

    # fit on training set
    train(trainloader, net, criterion, optimizer, num_epochs, board=board, testloader=testloader,
          scheduler_mode=scheduler_mode, scheduler=scheduler)

    # eval on test set
    eval_classifier_test_acc(testloader, net)

    # save model
    model_fname = model_id + MODEL_FILE_EXT
    model_fpath = os.path.join(TORCH_MODEL_DIR, model_fname)
    torch.save(net.state_dict(), model_fpath)


def train_manager(model_type: str,
                  model_id: str,
                  example_idxs: Optional[List[int]] = None,
                  train_opts: Optional[dict] = None):
    if model_type == 'NNConvResNetRGB':
        _, _, trainloader, testloader, _ = get_cifar10_data(
            batch_size=BATCH_SIZE_CIFAR10,
            example_idxs_train=example_idxs
        )
        train_on_cifar10(trainloader, testloader, model_id, train_opts_spec=train_opts)
        insert_model_info(model_type, model_id, example_idxs, train_opts)

def insert_model_info(model_type: str,
                      model_id: str,
                      example_idxs: Optional[List[int]] = None,
                      train_opts: Optional[dict] = None):
    new_model_info = dict(
        model_type=model_type,
        model_id=model_id,
        example_idxs=example_idxs,
        train_opts=train_opts,
        filename=model_id + '.pth'
    )
    update_model_info_file('insert', new_model_info)



if __name__ == '__main__':
    _, _, trainloader, testloader, _ = get_cifar10_data(
        batch_size=BATCH_SIZE_CIFAR10,
        example_idxs_train=[0, 700]
    )
    model_id = str(int(time.time() * 1e6))
    train_opts = dict(num_epochs=1)
    train_on_cifar10(trainloader, testloader, model_id, train_opts=train_opts)
    insert_model_info("NNConvResNetRGB", model_id)
