"""
Flask app for controlling ML operations

Run at command line:
    FLASK_HOST=10.0.0.34 FLASK_PORT=48513 HOME_DIR=/home/nuc/docker_exps_home_dir/ python src/docker_exps/app/app.py

Example to post data:
    curl -X POST http://10.0.0.34:48513/get_example -H 'Content-Type: application/json' -d '{"example_id":5}'
"""

from typing import Union, List
import os
from pprint import pprint

import numpy as np
from flask import Flask, jsonify, request
import requests

from ossr_utils.io_utils import save_json, load_json

from src.docker_exps.utils.app_utils import is_range_list, is_valid_model_type, is_existing_model_id, preprocess_train_route_args
from src.docker_exps.constants import CIFAR10_CLASSES, DATASET_NAMES, MODEL_INFO_FPATH, \
    DATASET_SUBSETS
from src.docker_exps.constants_network import URL_TRAIN_INTERNAL
from src.docker_exps.ml.ml_ops import get_cifar10_data
from src.docker_exps.ml.predict import predict_manager


CONTROLLER_STATUS = dict(
    train_jobs=[],
    predict_jobs=[],
    nodes=dict(
        train=[],
        pred=[]
    )
)


# init model info file if it doesn't exist
if not os.path.exists(MODEL_INFO_FPATH):
    save_json(MODEL_INFO_FPATH, {})


app = Flask(__name__)

@app.route('/')
def home():
    return '<h3>Hello. The app is running.</h3>'

@app.route('/status')
def status():
    return jsonify(CONTROLLER_STATUS)

@app.route('/get_examples', methods=['POST'])
def get_example():
    req_info = request.get_json(force=True)
    example_idxs: Union[list, int] = req_info['example_idxs'] # single int or list of index endpoints
    data_subset: str = req_info['dataset_type']

    pprint(req_info)

    try:
        assert data_subset in DATASET_SUBSETS
    except:
        return jsonify(dict(exception='Specify valid data_subset.'))

    if isinstance(example_idxs, int):
        ex_mode = 'int'
    elif is_range_list(example_idxs):
        ex_mode = 'list'
    else:
        return jsonify(dict(exception='Specify valid example_idxs.'))

    if ex_mode == 'int':
        ex_idxs = [example_idxs]
    else:
        ex_idxs = list(range(example_idxs[0], example_idxs[1]))

    if data_subset == 'train':
        dataset = get_cifar10_data(train=True, test=False)[0]
    if data_subset == 'test':
        dataset = get_cifar10_data(train=False, test=True)[1]

    class_idxs: List[int] = []
    class_tags: List[str] = []
    examples: List[list] = []
    for i in ex_idxs:
        example_, class_id_ = dataset[i]
        class_idxs.append(class_id_)
        class_tags.append(CIFAR10_CLASSES[class_id_])
        examples.append(example_.tolist())

    res = dict(class_idxs=class_idxs, class_tags=class_tags, examples=examples)
    return jsonify(res)

@app.route('/train', methods=['POST'])
def train():
    req_info = request.get_json(force=True)
    req_info = preprocess_train_route_args(req_info)
    if 'exception' in req_info:
        return jsonify(req_info)
    pprint(req_info)
    res = requests.post(URL_TRAIN_INTERNAL + 'train', json=req_info)
    dictFromServer = res.json()
    return jsonify(dictFromServer)

@app.route('/predict', methods=['POST'])
def predict():
    req_info = request.get_json(force=True)
    model_id: str = req_info['model_id']
    examples: list = req_info['examples']

    temp = dict(model_id=model_id)
    temp['examples.shape'] = np.array(examples).shape
    pprint(temp)

    # TODO: spin up predict job
    pred_ids = predict_manager(model_id, examples)
    pred_tags = [CIFAR10_CLASSES[i] for i in pred_ids]

    res = dict(class_idxs=pred_ids, class_tags=pred_tags)
    return jsonify(res)

@app.route('/get_dataset_info', methods=['POST'])
def get_dataset_stats():
    req_info = request.get_json(force=True)
    dataset_name: str = req_info.get('dataset_name')

    try:
        assert dataset_name in DATASET_NAMES
    except:
        return jsonify(dict(exception='The dataset_name argument is invalid.'))

    if dataset_name == 'CIFAR10':
        trainset, testset = get_cifar10_data()[:2]
        ex, _ = trainset[0]

        res = dict(
            num_train=len(trainset),
            num_test=len(testset),
            ex_shape=ex.shape,
            num_classes=len(np.unique(trainset.targets))
        )
        return jsonify(res)

@app.route('/get_model_info')
def get_model_info():
    model_info = load_json(MODEL_INFO_FPATH)
    return jsonify(model_info)



if __name__ == "__main__":
    FLASK_HOST = os.environ['FLASK_CONTROLLER_HOST']
    FLASK_PORT = int(os.environ['FLASK_CONTROLLER_PORT'])
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)

