"""
This file contains common code for all task
"""
import logging
import sys

log = logging.getLogger(__name__)


def data_preprocessing():
    log.info("Starting data preprocessing ..")
    return None, None, None


class Model:
    """
    This is a general interface for a task model
    """

    def __init__(self, task_name):
        self.task_name = task_name
        log.info(f"Creating task {self.task_name} preprocessing ..")

    def train(self):
        log.info(f"Starting task {self.task_name} training ..")

    def test(self):
        log.info(f"Starting task {self.task_name} testing ..")

    def cleanup(self):
        log.info(f"Cleaning model for task {self.task_name} ..")


_log_root = logging.getLogger()
_log_root.setLevel(logging.DEBUG)

_log_ch = logging.StreamHandler(sys.stdout)
_log_ch.setLevel(logging.DEBUG)
_formatter = logging.Formatter('[%(asctime)s][%(name)s.%(funcName)s][%(levelname)s] %(message)s')
_log_ch.setFormatter(_formatter)
_log_root.addHandler(_log_ch)
