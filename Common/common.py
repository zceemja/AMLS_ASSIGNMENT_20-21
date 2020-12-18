"""
This file contains common code for all task
"""
import logging
import csv
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split

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
        self.model = None

        self.log.info("Preparing dataset..")
        X, y = self.prepare_data(dataset_dir, label_file)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def map_labels(self, rows):
        """
        Extracts labels from csv and maps them to dict as {img_basename: label_value}
        """
        raise NotImplementedError("Method read_labels not defined in task " + self.task_name)

    @staticmethod
    def read_csv(filename, delimiter='\t'):
        with open(filename, 'r') as label_csv:
            for row in csv.reader(label_csv, delimiter=delimiter):
                yield row

    def tune_model(self):
        """
        This function is used to select best model/optimise hyper-parameters
        """
        raise NotImplementedError("Method tune_model not defined in task " + self.task_name)

    def prepare_data(self, images_dir, labels_file, train=True):
        """
        Loads and preprocesses images and labels
        :param images_dir: directory of images
        :param labels_file: label csv file
        :param train: is this is training or testing data. If training, some data will be cached
        :return: dataset X and Y
        """
        raise NotImplementedError("Method prepare_data not defined in task " + self.task_name)

    def train(self):
        self.log.info(f"Starting task {self.task_name} training ..")
        if self.model is None:
            raise RuntimeError("Model must be selected before training")
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        score = metrics.accuracy_score(self.y_test, y_pred)
        self.log.info(f"Task {self.task_name} model achieved {score * 100:.3}% accuracy with train data")
        return score

    def test(self, dataset_dir, label_file):
        """
        Test model with completely new data
        """
        self.log.info(f"Starting task {self.task_name} testing ..")
        if self.model is None:
            raise RuntimeError("Model must be selected and trained before testing")
        X, y = self.prepare_data(dataset_dir, label_file, train=False)
        y_pred = self.model.predict(X)
        score = metrics.accuracy_score(y, y_pred)
        self.log.info(f"Task {self.task_name} model achieved {score * 100:.3}% accuracy with test data")
        return score

    def cleanup(self):
        self.log.info(f"Cleaning model for task ..")
        # In case model has been setup from before
        tf.keras.backend.clear_session()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
