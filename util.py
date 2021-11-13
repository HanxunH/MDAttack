import logging
import os
import json


def setup_logger(name, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)
    return logger


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def save_json(payload, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(payload, outfile)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Code imported from
    https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
