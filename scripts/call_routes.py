
import os
import requests
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt


# URL_BASE = 'http://' + os.environ['FLASK_HOST'] + ':' + os.environ['FLASK_PORT']
URL_BASE = 'http://10.0.0.34:48513/'


def main():
    # status
    res = requests.get(URL_BASE + 'status').json()
    pprint(res)

    # get example
    dictToSend = dict(
        example_idxs=[3, 6]
    )
    res = requests.post(URL_BASE + 'get_examples', json=dictToSend)
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
        model_id='5a0s8ydf0a9s8dyf',
        example=examples[:1].tolist()
    )
    res = requests.post(URL_BASE + 'predict', json=dictToSend)
    dictFromServer = res.json()
    pprint(dictFromServer)



if __name__ == '__main__':
    main()