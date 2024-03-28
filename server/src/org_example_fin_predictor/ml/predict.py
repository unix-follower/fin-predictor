"""
Loads the model from local file system and holds a reference.
"""

import keras
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error

from org_example_fin_predictor.config import app_config

_LSTM_MODEL = None
_GRU_MODEL = None


def _get_lstm_model():
    """
    Lazily init LSTM
    """
    global _LSTM_MODEL  # pylint: disable=global-statement
    if _LSTM_MODEL is None:
        _LSTM_MODEL = load_model(app_config.lstm_model_path)
    return _LSTM_MODEL


def _get_gru_model():
    """
    Lazily init GRU
    """
    global _GRU_MODEL  # pylint: disable=global-statement
    if _GRU_MODEL is None:
        _GRU_MODEL = load_model(app_config.gru_model_path)
    return _GRU_MODEL


def _get_model(model_type: str = None) -> keras.Model:
    """
    Choose a GRU or LSTM model. Default is GRU.
    """
    if model_type is None:
        model_type = ""

    match model_type.upper():
        case "LSTM":
            model = _get_lstm_model()
        case "GRU":
            model = _get_gru_model()
        case _:
            model = _get_gru_model()

    return model


def predict(data: np.ndarray, model_type: str = None) -> float:
    """
    Predict the price.
    :return: root mean squared error.
    """
    model = _get_model(model_type)
    prediction = model.predict(data)
    return np.sqrt(mean_squared_error(data, prediction))
