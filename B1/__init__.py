"""
Multiclass tasks (cartoon_set dataset)
B1: Face shape recognition: 5 types of face shapes
"""
import common


class Model(common.Model):
    def __init__(self, dataset_dir, label_file):
        super(Model, self).__init__("B1", dataset_dir, label_file)
