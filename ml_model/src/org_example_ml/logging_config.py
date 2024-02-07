"""
Documentation:
https://docs.python.org/3/library/logging.config.html
"""

from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s [%(threadName)s] %(filename)s: %(message)s",
            }
        },
        "root": {"level": "INFO"},
    }
)
