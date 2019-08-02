from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from hrc_discrim_learning.base import AdaptiveContext

import numpy as np
import joblib
from os import path

# TODO: implement incremental learning
# for now, just training in batch

class PerceptronLearner:
    def __init__(self):
        # TODO: tune params
        self.ppn = Perceptron(n_iter_no_change=10, eta0=0.1, random_state=0)
        self.sc = StandardScaler()
        self.type = "perceptron_incremental"

    def train(self, X, y):
        X_std = self.sc.fit_transform(X)
        self.ppn.fit(X_std, y)

    def predict(self, x):
        x_std = self.sc.transform(x)
        y_pred = self.ppn.predict(x_std)
        return y_pred

    def score(self, y, y_pred):
        print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    def save_models(self, dest_folder):
        mpath = path.join(dest_folder, self.type + "_select_features.pkl")
        joblib.dump(self.ppn, mpath)

        scaler_path = path.join(dest_folder, self.type + "_scaler.pkl")
        joblib.dump(self.sc, scaler_path)

    def load_models(self, dest_folder):
        mpath = path.join(dest_folder, self.type + "_select_features.pkl")
        self.ppn = joblib.load(mpath)

        scaler_path = path.join(dest_folder, self.type + "_scaler.pkl")
        self.sc = joblib.load(scaler_path)

class IncrementalFeatureSelector:
    def __init__(self, spatial_model, rank=None):
        self.learner = PerceptronLearner()
        self.type = "perceptron_feat_selection"

        # rank is a list of features
        # this is arbitrary right now
        self.rank = [
            "type",
            "size_relative",
            "color",
            "location"
        ]

        # TODO: update
        self.input_mapping = {
            "color": 0,
            "size_relative" : 1,
            "type" : 2,
            "location" : 3
            # "p_loc"    : 4,
            # "d_loc"    : 5
        }

        # self.output_mapping = {
        #     "none" : 0,
        #     "color" : 1,
        #     "size"  : 2,
        #     "type"  : 3,
        #     "location" : 4
        # }

        self.x_len = 4
        self.sp_model = spatial_model

    def calc_discrim_score(self, obj, context, feature):
        value = context.get_obj_context_value(obj, feature)
        orig_size = context.env_size
        remaining_objs, count = context.shared_features(feature, value)

        return count / orig_size, remaining_objs

    def produce_input(self, obj, context, feature):
        # TODO: update
        x = np.zeros(self.x_len)

        dscore, remaining_objs = self.calc_discrim_score(obj, context, feature)
        i = self.input_mapping[feature]
        x[i] = dscore
        return x, remaining_objs

    def preprocess(self, corpus_dict):
        X = []
        Y = []

        print("Corpus:")
        print(corpus_dict)

        for context in corpus_dict:
            context.init_spatial_model(self.sp_model)

            for obj, utt in corpus_dict[context]:
                working_set = context
                features_used = set([t[0] for t in utt])

                for f in self.rank:
                    x, remaining_objs = self.produce_input(obj, working_set, f)
                    y = (f in features_used)

                    if y:
                        # debug
                        if not remaining_objs:
                            # print("In preprocess:")
                            # print(utt, "selected: ", f)
                            # print("prev context")
                            # for obj in working_set.env:
                            #     print(obj.features)
                            pass
                        working_set = AdaptiveContext(remaining_objs)
                        working_set.init_spatial_model(self.sp_model)

                    X.append(x)
                    Y.append(y)

        return X, Y

    def train(self, X, Y):
        self.learner.train(X, Y)

    def _incremental_predict(self, obj, context, rank):
        for feature in rank:
            x, remaining_objs = self.produce_input(obj, context, feature)
            y = self.learner.predict([x])

            if y:
                context = AdaptiveContext(remaining_objs)
                context.init_spatial_model(self.sp_model)

                return feature, context
        return None, None

    def predict(self, obj, context):
        output = ""
        rank = self.rank

        context.init_spatial_model(self.sp_model)

        while True:
            feature, new_context = self._incremental_predict(obj, context, rank)

            # TODO: edge case
            if not feature:
                return output
            else:
                output += (context.get_obj_context_value(obj, feature) + " ")
                context = new_context

            rank = list(set(rank) - set([feature]))
            if not rank:
                return output

    def print_function(self):
        pass

    def save_models(self, destination):
        self.learner.save_models(destination)
    def load_models(self, destination):
        self.learner.load_models(destination)
