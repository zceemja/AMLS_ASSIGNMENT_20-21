"""
Binary task (celeba dataset)
A2: Emotion detection: smiling or not smiling.
"""
import common


class Model(common.Model):
    def __init__(self, dataset_dir, label_file):
        super(Model, self).__init__("A2", dataset_dir, label_file)