import logging
import logging.config
import os

import yaml


def setup_logger():
    config = yaml.safe_load(
        open(os.path.join(os.path.dirname(__file__), "../logging_config.yml"))
    )
    logging.config.dictConfig(config)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
