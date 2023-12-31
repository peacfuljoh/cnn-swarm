
import requests
from pprint import pprint
import time
from threading import Thread

import numpy as np
import matplotlib.pyplot as plt



URL_BASE = 'http://127.0.0.1'
PORT_CONTROLLER = 48515
PORT_TRAIN = 48516

URL_CONTROLLER = URL_BASE + ':' + str(PORT_CONTROLLER) + '/'
URL_TRAIN = URL_BASE + ':' + str(PORT_TRAIN) + '/'



def main_predict():
    # status
    res = requests.get(URL_CONTROLLER + 'status').json()
    pprint(res)

    # get example
    dictToSend = dict(
        example_idxs=[103, 110],
        dataset_type='test'
    )
    res = requests.post(URL_CONTROLLER + 'get_examples', json=dictToSend)
    dictFromServer = res.json()
    examples = np.array(dictFromServer['examples'])
    temp = {key: val for key, val in dictFromServer.items() if key != 'examples'}
    temp['examples.shape'] = examples.shape
    pprint(temp)

    if 0:
        for i in range(len(examples)):
            plt.figure(figsize=(2, 2))
            plt.imshow(np.transpose(examples[i], (1, 2, 0)))
        plt.show()

    # predict on example
    dictToSend = dict(
        model_id='cifar10_003',
        examples=examples.tolist()
    )
    res = requests.post(URL_BASE + 'predict', json=dictToSend)
    dictFromServer = res.json()
    pprint(dictFromServer)

def main_train():
    # status
    def check_status():
        while 1:
            try:
                res = requests.get(URL_CONTROLLER + 'status').json()
            except:
                res = {}
            pprint(res)
            # if (len(res['jobs']) > 0) and all([r['complete'] for r in res['jobs'].values()]):
            #     break
            time.sleep(10)
    check_status_thread = Thread(target=check_status)
    check_status_thread.start()

    # train
    def start_job():
        model_id = str(int(time.time() * 1e6))
        # model_id = '1687044056050585'
        dictToSend = dict(
            model_type="NNConvResNetRGB",
            model_id=model_id,
            example_idxs=[0, 1000],
            train_opts=dict(num_epochs=2)
        )
        try:
            res = requests.post(URL_CONTROLLER + 'train', json=dictToSend)
            dictFromServer = res.json()
        except:
            dictFromServer = {}
        pprint(dictFromServer)


    if 1:
        start_job()
        for i in range(3):
            time.sleep(30)
            start_job()

def main_model_info():
    res = requests.get(URL_CONTROLLER + 'get_model_info').json()
    pprint(res)




if __name__ == '__main__':
    # main_predict()
    main_train()
    # main_model_info()
