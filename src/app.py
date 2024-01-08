# noinspection PyUnresolvedReferences
import src.logging_config

import sys
import traceback

from flask import Flask, json

from src.util import force_json_content_type_headers
from src.api import predict_controller

app = Flask(__name__)
app.register_blueprint(predict_controller.predict_api_v1)

NOT_FOUND_STATUS_CODE = 404
INTERNAL_SERVER_ERROR_STATUS_CODE = 500


@app.errorhandler(NOT_FOUND_STATUS_CODE)
def handle_not_found(_):
    response_body = {
        "message": "The resource is not found"
    }
    return json.dumps(response_body), NOT_FOUND_STATUS_CODE, force_json_content_type_headers


@app.errorhandler(Exception)
def handle_internal_server_error(e):
    etype, value, tb = sys.exc_info()
    app.logger.error(traceback.print_exception(etype, value, tb))
    app.logger.error(e)
    response_body = {
        "message": "An error has occurred"
    }
    return response_body, INTERNAL_SERVER_ERROR_STATUS_CODE
