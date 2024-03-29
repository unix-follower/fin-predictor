from typing import Callable
import numpy as np
from flask_healthz import HealthError
from org_example_fin_predictor.config import app_config
from org_example_fin_predictor.util import constants


def liveness():
    prices = np.array(
        [164.36000061035156, 166.50999450683594, 166.47000122070312, 167.64999389648438]
    )
    prices = np.ndarray(shape=(1, 1), buffer=prices)

    model = app_config.get_model(constants.GRU)
    model.predict(prices)

    model = app_config.get_model(constants.LSTM)
    model.predict(prices)


_MODEL_LOAD_FAILED_ERR_MSG = "model has failed to load"


def readiness():
    _run_check(_check_gru_model_loads, f"GRU {_MODEL_LOAD_FAILED_ERR_MSG}")
    _run_check(_check_gru_model_loads, f"LSTM {_MODEL_LOAD_FAILED_ERR_MSG}")


def _run_check(fn: Callable[[], None], failure_message: str):
    try:
        fn()
    except Exception as e:
        raise HealthError(failure_message) from e


def _check_gru_model_loads():
    app_config.get_model(constants.GRU)


def _check_lstm_model_loads():
    app_config.get_model(constants.LSTM)
