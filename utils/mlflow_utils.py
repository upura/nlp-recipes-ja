import mlflow
from omegaconf import DictConfig


def log_all_params(params):
    """dictの中にdictが入っている場合も書き出す"""
    for k, v in params.items():
        if isinstance(v, (dict, DictConfig)):
            log_all_params({f"{k._k}": _v for _k, _v in v.items()})
        else:
            mlflow.log_param(k, v)
