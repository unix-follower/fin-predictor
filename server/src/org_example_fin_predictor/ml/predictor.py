import numpy as np
from sklearn.metrics import mean_squared_error

from org_example_fin_predictor.config import app_config


def predict(data: np.ndarray, model_type: str = None) -> float:
    """
    Predict the price.
    :return: root mean squared error.
    """
    model = app_config.get_model(model_type)
    prediction = model.predict(data)
    return np.sqrt(mean_squared_error(data, prediction))
