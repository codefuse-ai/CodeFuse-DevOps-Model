import sys
import logging
from transformers.utils import logging as t_logging

from loguru import logger

class LoggerHandler(logging.Handler):

    def __init__(self):
        super().__init__()
        self.log = ""

    def reset(self):
        self.log = ""

    def emit(self, record):
        if record.name == "httpx":
            return
        log_entry = self.format(record)
        self.log += log_entry
        self.log += "\n\n"


def reset_logging():
    r"""
    Removes basic config of root logger
    """
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))


# def get_logger(name: str) -> logging.Logger:
#     formatter = logging.Formatter(
#         fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S"
#     )
#     handler = logging.StreamHandler(sys.stdout)
#     handler.setFormatter(formatter)

#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#     logger.addHandler(handler)

#     return logger

# def get_logger(name: str):
#     return logger

def get_logger(name: str):
    logger = t_logging.get_logger('transformers')
    return logger