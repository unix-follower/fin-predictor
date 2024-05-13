import json
import logging

_log_fields = {
    "name",
    "msg",
    "levelname",
    "pathname",
    "module",
    "lineno",
    "threadName",
    "process",
    "metric",
    "mtype"
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {key: record.__dict__.get(key) for key in _log_fields}
        return json.dumps(log_data)
