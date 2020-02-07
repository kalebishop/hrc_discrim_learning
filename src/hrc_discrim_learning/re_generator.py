#!/usr/bin/env python
# TODO uncomment
# from sklearn.svm import SVR
import numpy as np
import copy
import pickle

from hrc_discrim_learning.speech_module import SpeechModule

class SpeechLearner:
    def __init__(self, label):
        # TODO tune
        self.label = label
        # self.clf = SVR(kernel='linear', C=100, gamma='auto', epsilon=.1)

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def plot_learned_function(self, data):
        # TODO implement
        raise NotImplementedError
        pass

    def save_model(self, filename=""):
        if not filename:
            filename = "/ros/catkin_ws/src/workspace_speech/models/speech/svm_" + self.label + ".pkl"
        pickle.dump(model, open(filename, 'wb'))

    def load_model(self, filename=""):
        if not filename:
            filename = "/ros/catkin_ws/src/workspace_speech/models/speech/svm_" + self.label + ".pkl"
        self.clf = pickle.load(open(filename, 'rb'))

class REG:
    def __init__(self):
        w2c = "data/w2c_4096.txt"
        self.sm = SpeechModule(w2c)
        self.theta = 0.5

        self.features = ["color", "size", "dimensions"]

        self.models = {"color": SpeechLearner("color"),
                        "size": SpeechLearner("size"),
                        "dimensions": SpeechLearner("dim")}

    def train_model(self, feature, x, y):
        model = self.modles[feature]
        model.train(x, y)

    def _generate_single_output(self, object, context, feature_set):
        pscore_dict = {}
        for feature in feature_set:
            label, score, data, kept_objects = self.get_model_input(feature, object, context)
            X = [score, data]
            # UNCOMMENT WHEN MODEL IS IN PLACE
            pscore = self.models[feature].predict([X])
            pscore_dict[(feature, label)] = pscore

        # TODO finish
        best_candidate = max(pscore_dict.keys(), key=lambda x: pscore_dict[x])
        if pscore_dict[best_candidate] >= self.theta:
            return best_candidate
        else:
            return None, None

    def generate_output(self, object, context):
        # context should include object
        # TODO finish with ML model
        output = ""

        # first: boil down to type
        kept_objects = []
        type = object.get_feature_val("type")
        for o in context.env:
            o_type = object.get_feature_val("type")
            if o_type == type:
                kept_objects.append(o)
        context = self.update_context(kept_objects)

        # next: iterate through possible features
        feature_set = copy.copy(self.features)
        while feature_set:
            feature, label = self._generate_single_output(object, context, feature_set)
            if not label:
                output.append("type")
                return output
            output.append(label + " ")

        return "ERR: check REGenerator"

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
