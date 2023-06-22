

from ossr_utils.io_utils import load_json

from src.docker_exps.constants import MODEL_TYPES, MODEL_INFO_FPATH


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


def preprocess_train_route_args(req_info: dict) -> dict:
    model_type: str = req_info['model_type']
    model_id: str = req_info['model_id']
    example_idxs = req_info.get('example_idxs')  # List[int]
    train_opts = req_info.get('train_opts')  # dict

    try:
        assert isinstance(model_type, str)
        assert is_valid_model_type(model_type)
        assert isinstance(model_id, str)
        assert not is_existing_model_id(model_id)
        assert example_idxs is None or is_range_list(example_idxs)
    except:
        return dict(exception='InvalidInputArg', content=req_info)

    # pprint(req_info)

    return dict(
        model_type=model_type,
        model_id=model_id,
        example_idxs=example_idxs,
        train_opts=train_opts
    )


