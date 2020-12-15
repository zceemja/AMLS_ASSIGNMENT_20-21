"""
This file contains common code for all task
"""
import logging
import sys
import os
import csv
import numpy as np
import tensorflow as tf
import cv2
from sklearn import metrics

log = logging.getLogger(__name__)


def data_preprocessing():
    log.info("Starting data preprocessing ..")
    return None, None, None


class Model:
    """
    This is a general interface for a task model
    """
    KFOLDS = 10

    def __init__(self, task_name, dataset_dir, label_file):
        self.log = logging.getLogger(task_name)
        self.log.info(f"Creating task preprocessing ..")
        self.task_name = task_name
        self.dataset_dir = dataset_dir
        self.images = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
        self.labels = []
        with open(label_file, 'r') as label_csv:
            for row in csv.reader(label_csv, delimiter='\t'):
                self.labels.append(row)

        # Setup common objects
        self.model = None
        self.X_train = np.array([])
        self.X_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])

    def tune_model(self):
        """
        This function is used to select best model/optimise hyperparameters
        """
        self.log.info(f"Tuning model ..")

    def load_images(self, image_dir):
        """
        Load and preprocess images
        """
        imgs = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        results = []
        for image_path in imgs:
            results.append(cv2.imread(image_path))
        return results

    def train(self):
        self.log.info(f"Starting task {self.task_name} training ..")
        if self.model is None:
            raise RuntimeError("Model must be selected before training")
        self.model.fit(self.X_train, self.y_train)

    def test(self):
        self.log.info(f"Starting task {self.task_name} testing ..")
        if self.model is None:
            raise RuntimeError("Model must be selected and trained before testing")
        y_pred = self.predict(self.X_test)
        score = metrics.accuracy_score(self.y_test, y_pred)
        self.log.info(f"Task {self.task_name} model achieved {score*100:.3}% accuracy")

    def predict(self, data):
        """
        Predict some samples on a model
        """
        if self.model is None:
            raise RuntimeError("Model must be selected and trained before predicting")
        return self.model.predict(data)

    def cleanup(self):
        self.log.info(f"Cleaning model for task ..")
        # In case model has been setup from before
        tf.keras.backend.clear_session()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None


_log_root = logging.getLogger()
_log_root.setLevel(logging.DEBUG)

_log_ch = logging.StreamHandler(sys.stdout)
_log_ch.setLevel(logging.DEBUG)
_formatter = logging.Formatter('[%(asctime)s][%(name)s.%(funcName)s][%(levelname)s] %(message)s')
_log_ch.setFormatter(_formatter)
_log_root.addHandler(_log_ch)
