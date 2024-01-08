import numpy as np
from flask import Blueprint, request, json

from src.ml import predict
from src.util import force_json_content_type_headers
from logging import getLogger

predict_api_v1 = Blueprint("predict_api_v1", __name__)

logger = getLogger(__name__)


@predict_api_v1.post("/api/v1/predict")
def make_prediction():
    json_body = request.get_json()
    data = np.ndarray((1, 1), buffer=np.array(json_body), dtype=float)
    prediction = predict.predict(data)
    response_body = {
        "prediction": prediction
    }
    logger.info(response_body)
    return json.dumps(response_body), force_json_content_type_headers
