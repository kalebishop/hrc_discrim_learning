#!/usr/bin/env python
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import pickle
import rospy

class SpeechLearner:
    def __init__(self, label):
        # TODO tune
        self.label = label
        self.clf = SVR(kernel='linear', C=100, gamma='auto', epsilon=.1)

    def train(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def plot_learned_function(self, data):
        # TODO implement
        raise NotImplementedError

    def save_model(self, filename="/home/kaleb/ros/catkin_ws/src/workspace_speech/models/speech/svm_" + self.label + ".pkl"):
        pickle.dump(model, open(filename, 'wb'))

    def load_model(self, filename="/home/kaleb/ros/catkin_ws/src/workspace_speech/models/speech/svm_" + self.label + ".pkl"):
        self.clf = pickle.load(open(filename, 'rb'))

class FullSpeechPredictor:
    def __init__(self, learner_dict, sp_model):
        self.models = learner_dict

    def produce_input(self, feature, object, context):
        orig_size = context.env_size
        new_objs, new_size = context.feature_match(feature, object.get_feature_val(feature))

        dscore_net = orig_size - new_size
        dscore_scaled = dscore_net / orig_size

        return dscore_net, dscore_scaled, new_objs

    def _incremental_predict(self, obj, context, features):
        for f in features:
            dscore_net, dscore_scaled, objs_remaining = self.produce_input(f, obj, context)

            Y = self.models[f].predict([[dscore_net, dscore_scaled]])[0]

        if Y:
            new_context = Context(objs_remaining, context.sp_clf)
            label = context.get_obj_label(obj, f)
            return f, label, new_context

        return None, None, None

    def produce_output(self, obj_id, context):
        obj = context.object_lookup(obj_id)

        # type is added as a given
        type = obj.get_feature_val("type")
        output = ""

        while True:
            feature, label, new_context = self._incremental_predict(obj, context, features)
            # TODO: finish!
            raise NotImplementedError
