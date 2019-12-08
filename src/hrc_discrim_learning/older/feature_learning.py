#!/usr/bin/env python
from hrc_discrim_learning.base import Object, AdaptiveContext
from hrc_discrim_learning.simple_listener import SimpleListener
from collections import defaultdict
import math

class IncrementalLearner:
    """
    Just a basic incremental algorithm for feature selection
    Training data determines rank
    """
    def __init__(self, spatial_model):
        # self.listener = SimpleListener()
        self.type = 'feature_incremental'
        self.spatial_model = spatial_model

    def preprocess(self, corpus_dict):
        # for the incremental algorithm, we don't actually need to store
        # any data on the context - just on the obj features and output
        X = []
        Y = []

        for context in corpus_dict:
            for obj, utt in corpus_dict[context]:

                X.append(obj.features)
                Y.append(utt)

        return X, Y

    def train(self, X, Y):
        # dict for usage of each feature
        usage = defaultdict(lambda: 0)

        # X is a list of feature dicts
        # Y is a list of (feature, term) tuple lists

        for x, y in zip(X, Y):
            for f, term in y:
                usage[f] += 1

        self.rank = sorted(usage.keys(), key=lambda x: usage[x], reverse=True)

    def predict_from_spatial(self, obj, context):
        working_set = context
        count = context.env_size
        sp_model = self.spatial_model
        working_set.init_spatial_model(sp_model)

        output = ""

        for feature in self.rank:
            value = working_set.get_obj_context_value(obj, feature)
            shared, new_count = working_set.shared_features(feature, value)

            if new_count < count:
                working_set = shared
                shared.init_spatial_model(sp_model)
                output += (' ' + value)
            if new_count == 1:
                return output

        return output

    def print_function(self):
        print(self.rank)
