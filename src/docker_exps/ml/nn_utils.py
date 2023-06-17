
from typing import Tuple, List, Union

import numpy as np

import torch
from torch.nn import functional as F


def get_test_probs_and_labels(model, dataloader):
    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    class_probs = []
    class_label = []
    with torch.no_grad():
        for data in dataloader:  # one batch at a time
            images, labels = data
            output = model(images)
            class_probs_batch = F.softmax(output, dim=1)
            # class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(labels)

    # test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_probs = torch.cat(class_probs)
    test_labels = torch.cat(class_label)

    return test_probs, test_labels


def select_n_random(data, labels, n=100):
    '''Selects n random datapoints and their corresponding labels from a dataset'''
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))[:n]
    return data[perm], labels[perm]


def save_model(model, path):
    torch.save(model.state_dict(), path)


def eval_classifier_test_acc(testloader,
                             net,
                             top_n: int = 3) \
        -> Tuple[float, float]:
    """Test"""
    correct = 0
    correct_top_n = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_top_n = torch.argsort(outputs.data, dim=1)[:, -top_n:]
            labels_col = torch.reshape(labels, (len(labels), 1))
            correct_top_n += torch.any(predicted_top_n == labels_col, dim=1).sum().item()

    print('Accuracy on %d test examples: %d %%' % (total, 100 * correct / total))
    print('Top-%d accuracy on %d test examples: %d %%' % (top_n, total, 100 * correct_top_n / total))

    return correct / total, correct_top_n / total

