import logging
from os import getenv


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(getenv("LOG_LEVEL", logging.INFO))
    # my custom format - see https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(funcName)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
