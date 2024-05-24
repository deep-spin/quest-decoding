import logging

def fix_loggers(name="transformers"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    