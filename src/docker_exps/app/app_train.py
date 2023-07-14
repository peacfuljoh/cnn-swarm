"""
Flask app for training ML models
"""

import os
from pprint import pprint
from threading import Thread

from flask import Flask, jsonify, request

from src.docker_exps.utils.app_utils import preprocess_train_route_args
from src.docker_exps.ml.train import train_manager
from src.docker_exps.constants_train import TRAIN_STATUS, JOBS_INIT_INFO, JOB_MSG_QUEUE, TRAIN_TASKS




def train_status_manager():
    while 1:
        msg = JOB_MSG_QUEUE.get() # dict(type=..., data=...)
        job_id = msg['data']['job_id']
        if msg['type'] == 'train_job_complete':
            TRAIN_STATUS['jobs'][job_id]['complete'] = True
            del TRAIN_TASKS[job_id] # TODO: enable this?
        if msg['type'] == 'train_job_update':
            TRAIN_STATUS['jobs'][job_id]['duration'] = msg['data']['duration']
            TRAIN_STATUS['jobs'][job_id]['progress'] = msg['data']['progress']

train_status_manager_thread = Thread(target=train_status_manager, daemon=True)
train_status_manager_thread.start()




app = Flask(__name__)

@app.route('/')
def home():
    return '<h3>Hello. The app is running.</h3>'

@app.route('/status')
def status():
    print(TRAIN_STATUS)
    return jsonify(TRAIN_STATUS)

@app.route('/type')
def type():
    return jsonify(dict(type='train'))

@app.route('/train', methods=['POST'])
def train():
    req_info = request.get_json(force=True)
    req_info = preprocess_train_route_args(req_info)
    if 'exception' in req_info:
        return jsonify(req_info)
    pprint(req_info)

    # spin up train job
    try:
        job_id = JOBS_INIT_INFO['id_latest']
        JOBS_INIT_INFO['id_latest'] += 1
        TRAIN_STATUS['jobs'][job_id] = dict(
            duration=0.0,
            progress=0.0,
            complete=False
        )
        train_job_args = (req_info['model_type'], req_info['model_id'], req_info['example_idxs'], req_info['train_opts'], job_id)
        train_task = Thread(target=train_manager, daemon=True, args=train_job_args)
        train_task.start()
        TRAIN_TASKS[job_id] = train_task
    except:
        res = dict(exception='FailedToStartTrainJob')
        return jsonify(res)

    res = dict(status=dict(initiated=True))
    return jsonify(res)



if __name__ == "__main__":
    FLASK_HOST = os.environ['FLASK_TRAIN_HOST']
    FLASK_PORT = int(os.environ['FLASK_TRAIN_PORT'])
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)
