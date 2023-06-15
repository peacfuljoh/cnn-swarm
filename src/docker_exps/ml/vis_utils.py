
from typing import Optional, Callable, Sequence, Dict, List
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from nn_utils import get_test_probs_and_labels, select_n_random


class DLViz():
    """
    Visualization class for tensor-format images. Lays out images in grid using torchvision make_grid() util.
    """
    def __init__(self,
                 dataloader: Optional[torch.utils.data.DataLoader] = None,
                 labels: Optional[Sequence[str]] = None,
                 data_func: Optional[Callable] = None):
        self._loader = copy.deepcopy(dataloader)
        self._labels = labels
        self._data_func = data_func

    def show_grid(self,
                  nrows: int,
                  ncols: int,
                  show: bool = False):
        # get some random training images
        dataiter = iter(self._loader)
        images_all = []
        labels_all = []
        for i in range(nrows):
            images_, labels_ = next(dataiter)
            images_all.append(images_)
            labels_all.append(labels_)

        # create grid from list of images
        self._create_grid_from_images(images_all, ncols)

        # print labels
        if show:
            self._print_labels(labels_all)

        # show composite image
        self._imshow(show=show)

    def _create_grid_from_images(self,
                                 images_all: List[torch.Tensor],
                                 nrow: int):
        images = torch.cat(images_all, dim=0)
        self._im_grid = torchvision.utils.make_grid(images, nrow=nrow)

    def _print_labels(self,
                      labels_all: List[str]):
        if self._labels is not None:
            for i, labels_ in enumerate(labels_all):
                print('\n' + ' '.join('%10s' % self._labels[lab] for lab in labels_.tolist()))

    def show_grid_from_tensor(self,
                              x: torch.Tensor,
                              nrows: int,
                              ncols: int,
                              show: bool = False):
        dims = x.shape
        images = [x] + [torch.zeros((nrows * ncols - dims[0], *dims[1:]))]
        self._create_grid_from_images(images, nrow=ncols)
        self._imshow(show=show)

    def _imshow(self, show: bool = False):
        if self._data_func is not None:
            img = self._data_func(self._im_grid)
        else:
            img = self._im_grid
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if show:
            plt.show()


class Tensorboard():
    """
    Class to manage tensorboard functionality.

    To initiate local process, run at command line:
        tensorboard --logdir=/home/nuc/torch_runs/
    """
    def __init__(self,
                 log_dir: str):
        self._writer = SummaryWriter(log_dir)

    def add_image(self,
                  img_name: str,
                  img: torch.Tensor):
        # add an image
        self._writer.add_image(img_name, img)
        self._writer.close()

    def add_graph(self,
                  model,
                  train_dataloader):
        # add visualization of network structure
        dataiter = iter(train_dataloader)
        images, labels = next(dataiter)
        self._writer.add_graph(model, images)
        self._writer.close()

    def add_embedding(self,
                      dataset,
                      classes: Dict[int, str]):
        """Visualize data using PCA or t-SNE"""
        # select random images and their target indices
        images, labels = select_n_random(dataset.data, dataset.targets)

        # get the class labels for each image
        class_labels = [classes[lab.item()] for lab in labels]

        # log embeddings
        features = images.view(-1, 28 * 28)
        self._writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
        self._writer.close()

    def add_scalar(self,
                   name: str,
                   value: float,
                   iter: int):
        self._writer.add_scalar(name, value, iter)
        self._writer.close()

    def add_pr_curves(self,
                      model,
                      dataloader,
                      classes: Dict[int, str],
                      epoch: int = 0):
        # get test probs and labels
        test_probs, test_label = get_test_probs_and_labels(model, dataloader)

        # plot all the pr curves
        for class_index in range(len(classes)):
            self._add_pr_curve_tensorboard(class_index, test_probs, test_label, classes, epoch)

    def _add_pr_curve_tensorboard(self,
                                  class_index: int,
                                  test_probs: torch.Tensor,
                                  test_labels: torch.Tensor,
                                  classes: Dict[int, str],
                                  epoch: int = 0):
        # takes in a "class_index" and plots the corresponding precision-recall curve
        tensorboard_truth = test_labels == class_index
        tensorboard_probs = test_probs[:, class_index]

        self._writer.add_pr_curve(classes[class_index], # label
                                  tensorboard_truth, # 2D indicator array
                                  tensorboard_probs, # class probs
                                  global_step=epoch)
        self._writer.close()