"""
Binary task (celeba dataset)
A1: Gender detection: male or female.
"""
import common
from A1 import face_detect


class Model(common.Model):
    def __init__(self, dataset_dir, label_file):
        super(Model, self).__init__("A1", dataset_dir, label_file)
        face_detect.detect_faces(self.images)
