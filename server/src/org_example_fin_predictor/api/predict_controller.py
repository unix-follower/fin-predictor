"""
REST API for making price predictions.
"""

from logging import getLogger

import numpy as np
from flask import Blueprint, json, request

from org_example_fin_predictor.ml import predictor
from org_example_fin_predictor.util import util

predict_api_v1 = Blueprint("predict_api_v1", __name__)

logger = getLogger("root")


@predict_api_v1.post("/api/v1/predict")
def make_prediction():
    """
    Take prices from json array input and make prediction.
    """
    json_body = request.get_json()
    prices = json_body["prices"]
    data = np.ndarray((1, 1), buffer=np.array(prices), dtype=float)
    prediction = predictor.predict(data)
    response_body = {"prediction": prediction}
    logger.info(response_body)
    return json.dumps(response_body), util.force_json_content_type_headers
