"""
This file contains common code for all task
"""
import logging
import sys
import os
import csv

log = logging.getLogger(__name__)


def data_preprocessing():
    log.info("Starting data preprocessing ..")
    return None, None, None


class Model:
    """
    This is a general interface for a task model
    """

    def __init__(self, task_name, dataset_dir, label_file):
        log.info(f"Creating task {task_name} preprocessing ..")
        self.task_name = task_name
        self.dataset_dir = dataset_dir
        self.images = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
        self.labels = []
        with open(label_file, 'r') as label_csv:
            for row in csv.reader(label_csv):
                self.labels.append(row)

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
