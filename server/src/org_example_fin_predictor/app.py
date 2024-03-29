"""
Flask app API definitions.
"""

import sys
import traceback

from flask import Flask, json
from flask_healthz import Healthz

from org_example_fin_predictor.api import predict_controller
from org_example_fin_predictor.util import util

app = Flask(__name__)
Healthz(app)
app.register_blueprint(predict_controller.predict_api_v1)
app.config.from_object("org_example_fin_predictor.config.app_config.Config")

NOT_FOUND_STATUS_CODE = 404
INTERNAL_SERVER_ERROR_STATUS_CODE = 500


@app.errorhandler(NOT_FOUND_STATUS_CODE)
def handle_not_found(_):
    """
    The handler for 404 errors.
    """
    response_body = {"message": "The resource is not found"}
    return json.dumps(response_body), NOT_FOUND_STATUS_CODE, util.force_json_content_type_headers


@app.errorhandler(Exception)
def handle_internal_server_error(e):
    """
    The handler for 500 errors.
    """
    etype, value, tb = sys.exc_info()
    app.logger.error(traceback.print_exception(etype, value, tb))
    app.logger.error(e)
    response_body = {"message": "An error has occurred"}
    return response_body, INTERNAL_SERVER_ERROR_STATUS_CODE
