"""
Multiclass tasks (cartoon_set dataset)
B2: Eye colour recognition: 5 types of eye colours.
"""
import common


class Model(common.Model):
    def __init__(self, dataset_dir, label_file):
        super(Model, self).__init__("B2", dataset_dir, label_file)
