
import os

from ossr_utils.io_utils import load_json

from src.docker_exps.constants import STATUS_FPATH, TORCH_MODEL_DIR, MODEL_TYPES, MODEL_FILE_EXT, MODEL_INFO_FPATH


def get_sys_stats(init: bool = False) -> dict:
    if init:
        stats = dict(
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

def is_range_list(example_idxs) -> bool:
    return isinstance(example_idxs, list) and len(example_idxs) == 2 and all(isinstance(ex, int) for ex in example_idxs)

def is_valid_model_type(model_type: str) -> bool:
    return model_type in MODEL_TYPES

def is_existing_model_id(model_id: str) -> bool:
    model_ids_all = get_model_ids()
    # print(model_ids_all)
    return model_id in model_ids_all

def get_model_ids():
    return list(load_json(MODEL_INFO_FPATH).keys())
