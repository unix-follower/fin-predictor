import json
import os
import pathlib
from logging.config import dictConfig

_current_working_directory = os.getcwd()
_log_config_file_path = pathlib.Path(
    f"{_current_working_directory}/org_example_fin_predictor/config/log-config.json"
)
with _log_config_file_path.open(encoding="utf-8") as file:
    dictConfig(json.load(file))

# pylint: disable=invalid-name
loglevel = "DEBUG"
workers = 2
timeout = 120
bind = "0.0.0.0:5000"
accesslog = "-"
errorlog = None

access_log_format = json.dumps({
    "remote_address": "%(h)s",
    "user_name": "%(u)s",
    "date": "%(t)s",
    "status": "%(s)s",
    "method": "%(m)s",
    "url_path": "%(U)s",
    "query_string": "%(q)s",
    "protocol": "%(H)s",
    "response_length": "%(B)s",
    "referer": "%(f)s",
    "user_agent": "%(a)s",
    "request_time_seconds": "%(L)s",
    "process_id": "%(p)s",
})
