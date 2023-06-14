"""Experimental Python entry point for Docker process"""

import os
from typing import List

import numpy as np

from ossr_utils.io_utils import load_pickle, save_pickle, load_json, save_json



DATA_DIR = os.environ['DATA_DIR']
DATA_DIR_SRC = os.path.join(DATA_DIR, 'src')
DATA_DIR_DST = os.path.join(DATA_DIR, 'dst')



def main():
    # get data file paths
    fnames = [fname for fname in os.listdir(DATA_DIR_SRC) if '.json' in fname]
    fpaths = [os.path.join(DATA_DIR_SRC, fname) for fname in fnames]

    # load data
    data: List[dict] = [load_json(fpath) for fpath in fpaths]

    # merge data
    data_merge = dict()
    for d in data:
        for key, val in d.items():
            if key not in data_merge.keys():
                data_merge[key] = []
            assert isinstance(d[key], list)
            data_merge[key] += d[key]

    # save merged data
    fpath_merged = os.path.join(DATA_DIR_DST, 'merged.json')
    save_json(fpath_merged, data_merge)


if __name__ == "__main__":
    main()