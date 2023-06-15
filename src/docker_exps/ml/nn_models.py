
from typing import Tuple, Callable

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F



def _init_weights_standard(module):
    if isinstance(module, nn.Linear):
        k = np.sqrt(1. / module.weight.size(1))  # 1 / sqrt(in_features)
        module.weight.data.uniform_(-k, k)
        # module.weight.data.normal_(0, k)
        if module.bias is not None:
            # module.bias.data.zero_()
            module.bias.data.uniform_(-k, k)
    if isinstance(module, nn.Conv2d):
        k = np.sqrt(
            1. / (module.weight.size(1) * module.weight.size(2) * module.weight.size(3)))  # 1 / C_in * prod(kern_size)
        module.weight.data.uniform_(-k, k)
        # module.weight.data.normal_(0, k)
        if module.bias is not None:
            # module.bias.data.zero_()
            module.bias.data.uniform_(-k, k)







"""Fashion MNIST"""
class NNLinearReLUStack(nn.Module):
    """Basic linear + ReLU stack for Fashion MNIST"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # vectorize each image, (N, 1, 28, 28) -> (N, 28*28)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NNConvFashionMNIST(nn.Module):
    """CNN for Fashion MNIST dataset (28x28 grayscale images)"""
    def __init__(self,
                 im_size: Tuple[int, int]):
        super(NNConvFashionMNIST, self).__init__()

        assert im_size[0] == im_size[1]

        # layer params
        conv_kern_dims = (5, 5)
        num_feat_maps = (1, 6, 16)
        self._num_feat_layers = len(conv_kern_dims)

        feat_map_dim = im_size[0]
        for i in range(self._num_feat_layers):
            feat_map_dim = (feat_map_dim - (conv_kern_dims[i] - 1)) // 2
        print(feat_map_dim)

        num_features = feat_map_dim ** 2

        # feature extractor
        modules = []
        for i in range(self._num_feat_layers):
            modules += [
                nn.Conv2d(num_feat_maps[i], num_feat_maps[i + 1], conv_kern_dims[i]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ]
        self.convs = nn.Sequential(*modules)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_feat_maps[-1] * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 90),
            nn.ReLU(),
            nn.Linear(90, 10)
        )

    def forward(self, x):
        x = self.convs(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x


"""ImageNet"""
class NNConvLeNet(nn.Module):
    """Classic 6x6 digit image classification CNN"""
    def __init__(self):
        super(NNConvLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NNConvLeNetRGB(nn.Module):
    """RGB variant of LeNet"""
    def __init__(self):
        super(NNConvLeNetRGB, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) # kernel_size, stride
        self.conv2 = nn.Conv2d(6, 16, 5)
        self._num_feats = 16 * 5 * 5
        self.fc1 = nn.Linear(self._num_feats, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.act = F.relu

        self.apply(_init_weights_standard)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        if 1:
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.fc3(x)
        else:
            feat_counts = [self._num_feats, 256, 128, 64]
            for i in range(len(feat_counts) - 1):
                x_ = nn.Linear(feat_counts[i], feat_counts[i + 1])(x)
                x_ = self.act(x_)
                x_ = nn.Linear(feat_counts[i + 1], feat_counts[i + 1])(x_)
                x_ = x_ + nn.Linear(feat_counts[i], feat_counts[i + 1])(x)
                x = self.act(x_)
            x = self.fc3(x)
        return x


class NNConvResNetRGB(nn.Module):
    """RGB variant of LeNet with residual skip layers"""
    def __init__(self):
        super(NNConvResNetRGB, self).__init__()

        self._use_gap = True # global average pooling instead of FC layers

        self._num_subnets = 3
        self._num_res_mods_in_blocks = [3] * self._num_subnets
        self._feat_map_dims = [32, 16, 8]
        self._num_fmaps = [16, 32, 64]
        num_outputs = 10

        self._num_conv_layers = 2 * np.sum(self._num_res_mods_in_blocks) + 2
        self._num_lin_layers = 3
        self._num_layers = self._num_conv_layers + self._num_lin_layers
        print(f'num_layers = {self._num_layers}')

        kern_size = 3
        padding = (kern_size - 1) // 2

        self.conv1a = nn.Conv2d(3, self._num_fmaps[0], kern_size, padding=padding)
        self.conv1b = nn.Conv2d(self._num_fmaps[0], self._num_fmaps[0], kern_size, padding=padding)
        self.conv2a = nn.Conv2d(self._num_fmaps[0], self._num_fmaps[1], kern_size, padding=padding, stride=2)
        self.conv2b = nn.Conv2d(self._num_fmaps[1], self._num_fmaps[1], kern_size, padding=padding)
        self.conv3a = nn.Conv2d(self._num_fmaps[1], self._num_fmaps[2], kern_size, padding=padding, stride=2)
        self.conv3b = nn.Conv2d(self._num_fmaps[2], self._num_fmaps[2], kern_size, padding=padding)

        self.batch_norm1 = nn.BatchNorm2d(self._num_fmaps[0])
        self.batch_norm2 = nn.BatchNorm2d(self._num_fmaps[1])
        self.batch_norm3 = nn.BatchNorm2d(self._num_fmaps[2])

        self.pool = nn.MaxPool2d(self._feat_map_dims[2])
        # self.pool = lambda x: torch.mean(x, dim=[2, 3]) # equivalent to global max pooling
        self.fc_gap = nn.Linear(self._num_fmaps[2], num_outputs)

        self._num_feats = self._num_fmaps[2] * self._feat_map_dims[2] ** 2

        self.fc1 = nn.Linear(self._num_feats, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_outputs)

        self.act = F.relu

        self.apply(_init_weights_standard)

    def forward(self, x):
        funcs = [self.conv1a, self.conv2a, self.conv3a]
        funcs2 = [self.batch_norm1, self.batch_norm2, self.batch_norm3]

        for i in range(self._num_subnets):
            f = funcs[i]
            f2 = funcs2[i]

            x = f(x)
            x = f2(x)
            x = self.act(x)
            for j in range(self._num_res_mods_in_blocks[i]):
                x = self._forward_resnet_block(x, i)
                x = self.act(x)

        if not self._use_gap:
            x = x.view(-1, self._num_feats)

            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            x = self.act(x)
            x = self.fc3(x)
        else:
            x = self.pool(x)
            x = x.view(-1, self._num_fmaps[2])
            x = self.fc_gap(x)

        return x

    def _forward_resnet_block(self,
                              x: torch.Tensor,
                              block_id: int) \
            -> torch.Tensor:
        funcs = [self.conv1b, self.conv2b, self.conv3b]
        funcs2 = [self.batch_norm1, self.batch_norm2, self.batch_norm3]
        f = funcs[block_id]
        f2 = funcs2[block_id]

        y = f(x)
        y = f2(y)
        y = self.act(y)
        y = f(y)
        y = f2(y)
        y = y + x

        return y

