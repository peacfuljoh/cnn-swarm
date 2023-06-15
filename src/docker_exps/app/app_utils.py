
from ossr_utils.io_utils import load_json

from src.docker_exps.constants import STATUS_FPATH


def get_sys_stats(init: bool = False) -> dict:
    if init:
        stats = dict(
            num_examples=0,
            train_jobs=[],
            predict_jobs=[],
            nodes=dict(
                train=[],
                pred=[]
            )
        )
    else:
        stats = load_json(STATUS_FPATH)
    return stats
