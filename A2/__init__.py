"""
Binary task (celeba dataset)
A2: Emotion detection: smiling or not smiling.
"""
import common


class Model(common.Model):
    def __init__(self):
        super(Model, self).__init__("A2")
