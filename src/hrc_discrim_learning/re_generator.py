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
        if feature == "color":
            score, kept_objects = self.elim_objects_color(context, label)
        else:
            score, kept_objects = self.elim_objects_gradable(context, feature, label, data)

        return label, score, data, kept_objects

    def elim_objects_color(self, context, label):
        # we want to eliminate everything that the term label can NOT apply to
        score = 0
        kept_objects = []
        for o in context.env:
            this_label, data = self.sm.label_feature(o, context, "color")
            if label == this_label:
                kept_objects.append(o)
            else:
                score += 1

        return score, kept_objects

    def elim_objects_gradable(self, context, feature, label, label_score):
        # we want to eliminate everything that the term can NOT apply to
        # that is, everything that the term fits LESS well than the target object
        score = 0
        kept_objects = []
        for o in context.env:
            this_label, data = self.sm.label_feature(o, context, feature)
            if this_label == label and data >= label_score:
                kept_objects.append(o)
            else:
                score += 1

        return score, kept_objects

    def update_context(self, kept_objects):
        return Context(kept_objects)
