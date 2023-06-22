from typing import Optional, List
import time

import numpy as np
import torch

from src.docker_exps.ml.nn_utils import eval_classifier_test_acc
from src.docker_exps.constants_train import JOB_MSG_QUEUE


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
          augment: bool = True,
          job_id: Optional[int] = None) \
        -> List[float]:
    """Train"""
    assert isinstance(num_epochs, int) and num_epochs > 1

    t0 = time.time()

    epoch_loss = []
    num_train = len(trainloader) * trainloader.batch_size
    n_update_board = int((num_train / trainloader.batch_size) / 10)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('\n===== Epoch ' + str(epoch) + ' =====')
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

            # run info
            num_exs = trainloader.batch_size * (i + 1) # seen so far
            num_batch_tot = epoch * num_train + num_exs
            num_epochs_frac = epoch + num_exs / num_train # fractional epoch

            # update tensorboard
            if i % n_update_board == (n_update_board - 1):
                # train and test stats
                if testloader is not None:
                    acc, acc_top3 = eval_classifier_test_acc(testloader, net, top_n=3)

                # board entries
                if board is not None:
                    board.add_scalar('training loss', running_loss / num_exs, num_batch_tot)
                    if testloader is not None:
                        board.add_scalar('test accuracy', acc, num_batch_tot)
                        # board.add_scalar('test top-3 accuracy', acc_top3, num_batch_tot)

            # update job status
            msg_ = dict(
                type='train_job_update',
                data=dict(
                    job_id=job_id,
                    duration=round(time.time() - t0, 2),
                    progress=round(num_epochs_frac / num_epochs * 100.0, 2)
                )
            )
            JOB_MSG_QUEUE.put(msg_)

        epoch_loss.append(running_loss)
        print("epoch %d, loss = %.3f" % (epoch + 1, epoch_loss[epoch]))

        if scheduler_mode in ['plateau']:
            scheduler.step(running_loss)
        else:
            scheduler.step()

    print('Finished Training')

    return epoch_loss
