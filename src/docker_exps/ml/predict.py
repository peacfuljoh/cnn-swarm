
from typing import List, Union
import os

import numpy as np

import torch

from ossr_utils.io_utils import load_json

from src.docker_exps.ml.nn_models import NNConvResNetRGB
from src.docker_exps.app.app_utils import is_existing_model_id
from src.docker_exps.constants import MODEL_INFO_FPATH, TORCH_MODEL_DIR


def predict_manager(model_id: str,
                    examples: list) \
        -> List[int]:
    # load model
    assert is_existing_model_id(model_id)
    model_info = load_json(MODEL_INFO_FPATH)[model_id]
    model_type = model_info['model_type']
    if model_type == 'NNConvResNetRGB':
        net = NNConvResNetRGB()
    model_fpath = os.path.join(TORCH_MODEL_DIR, model_info['filename'])
    net.load_state_dict(torch.load(model_fpath))

    # perform prediction
    predicted = predict(net, torch.Tensor(examples))
    pred_ids: List[int] = predicted.tolist()

    return pred_ids

def predict(net,
            examples: torch.Tensor) \
        -> torch.Tensor:
    """Perform prediction on examples using specified network. Output type matches 'examples' input type."""
    if 0:
        import torchvision
        from src.docker_exps.ml.ml_ops import get_cifar10_data
        trainset, testset, trainloader, testloader, data_func = get_cifar10_data(batch_size=3, shuffle=False)
        examples, _ = next(iter(testloader))

    # print(examples.shape)
    # print(type(examples))
    # print(examples[:, 0, 0, 0])

    outputs = net(examples)
    predicted: torch.Tensor = torch.max(outputs.data, 1)[1]

    return predicted