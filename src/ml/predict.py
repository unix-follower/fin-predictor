import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

import src.util as util

_lstm_model = None
_gru_model = None


def _get_lstm_model():
    global _lstm_model
    if _lstm_model is None:
        _lstm_model = tf.keras.models.load_model(f"{util.models_directory}/lstm_model.keras")
    return _lstm_model


def _get_gru_model():
    global _gru_model
    if _gru_model is None:
        _gru_model = tf.keras.models.load_model(f"{util.models_directory}/gru_model.keras")
    return _gru_model


def _get_model(model_type: str = None):
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


def predict(data: np.ndarray, model_type: str = None) -> [float]:
    model = _get_model(model_type)
    prediction = model.predict(data)
    rmse = np.sqrt(mean_squared_error(data, prediction))
    return rmse
