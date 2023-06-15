"""
Flask app

Run at command line:
    FLASK_HOST=10.0.0.34 FLASK_PORT=48513 HOME_DIR=/home/nuc/docker_exps_home_dir/ python src/docker_exps/app/app.py

Example to post data:
    curl -X POST http://10.0.0.34:48513/get_example -H 'Content-Type: application/json' -d '{"example_id":5}'
"""

from typing import Union, List
import os
from pprint import pprint

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request

from ossr_utils.io_utils import save_json

from app_utils import get_sys_stats
from src.docker_exps.constants import CIFAR10_CLASSES, HOME_DIR, FLASK_HOST, FLASK_PORT, STATUS_FPATH
from src.docker_exps.ml.ml_ops import get_cifar10_data


# init status file
save_json(STATUS_FPATH, get_sys_stats(init=True))


app = Flask(__name__)

@app.route('/')
def home():
    return '<h3>Hello. The app is running.</h3>'

@app.route('/status')
def stats():
    res = get_sys_stats()
    return jsonify(res)

@app.route('/get_examples', methods=['POST'])
def get_example():
    req_info = request.get_json(force=True)
    example_idxs: Union[list, int] = req_info['example_idxs'] # single int or list of index endpoints

    pprint(req_info)

    if isinstance(example_idxs, int):
        ex_mode = 'int'
    elif isinstance(example_idxs, list) and len(example_idxs) == 2 and all(isinstance(ex, int) for ex in example_idxs):
        ex_mode = 'list'
    else:
        return jsonify(dict(exception='Specify valid example_idxs.'))

    trainset = get_cifar10_data(train=True, test=False)[0]

    if ex_mode == 'int':
        ex_idxs = [example_idxs]
    else:
        ex_idxs = list(range(example_idxs[0], example_idxs[1]))

    class_idxs: List[int] = []
    class_tags: List[str] = []
    examples: List[list] = []
    for i in ex_idxs:
        example_, class_id_ = trainset[i]
        class_idxs.append(class_id_)
        class_tags.append(CIFAR10_CLASSES[class_id_])
        examples.append(example_.tolist())

    res = dict(idxs=class_idxs, tags=class_tags, examples=examples)
    return jsonify(res)

@app.route('/train', methods=['POST'])
def train():
    req_info = request.get_json(force=True)
    model_type: str = req_info['model_type']
    train_data_exs: int = req_info['num_exs']

    pprint(req_info)

    # TODO: spin up train job

    res = dict(status='initiated')
    return jsonify(res)

@app.route('/predict', methods=['POST'])
def predict():
    req_info = request.get_json(force=True)
    model_id: str = req_info['model_id']
    example: np.ndarray = np.array(req_info['example'])

    temp = dict(model_id=model_id)
    temp['examples.shape'] = example.shape
    pprint(temp)

    # TODO: spin up predict job
    pred_id = 0
    pred_tag = CIFAR10_CLASSES[pred_id]

    res = dict(idxs=[pred_id], tags=[pred_tag])
    return jsonify(res)


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
