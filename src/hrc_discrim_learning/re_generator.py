#!/usr/bin/env python
from sklearn.svm import SVR
import numpy as np

from hrc_discrim_learning.speech_module import SpeechModule

class REG:
    def __init__(self):
        w2c = "data/w2c_4096.txt"
        self.sm = SpeechModule(w2c)

    def get_model_input(self, feature, object, context):
        # context should include object
        label, data = self.sm.label_feature(object, context, feature)

        # TODO finish
        raise NotImplementedError

    def elim_objects_color(self, object, context):
        # TODO implement
        # we want to eliminate everything that the term can NOT apply to
        raise NotImplementedError
        return score, kept_objects

    def elim_objects_gradable(self, context, feature, label):
        # TODO implement
        # we want to eliminate everything that the term can NOT apply to
        raise NotImplementedError
        return score, kept_objects

    def update_context(self, kept_objects):
        return Context(kept_objects)
