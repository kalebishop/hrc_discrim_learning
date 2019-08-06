from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from hrc_discrim_learning.base import AdaptiveContext

class SGD:
    def __init__(self):
        self.clf = SGDClassifier(max_iter = 10**5, loss='perceptron', penalty='l1')
        self.sc = StandardScaler()
        self.type = 'SGDSingle'

    def train(self, X, y):
        # training in large batches -
        # increment partial-fit online training later
        X_std = self.sc.fit_transform(X)
        self.clf.fit(X_std, y)

    def predict(self, X):
        X_std = self.sc.transform(X)
        y_pred = self.clf.predict(X_std)
        return y_pred

    def score(self, X, Y):
        raise NotImplementedError

    def get_learned_function(self):
        return self.clf.coef_, self.clf.intercept_

class IncrementalFeatureSelector:
    def __init__(self, features_ranked, spatial_model, size_model, color_model):
        # these all should be pretrained
        self.sp_model = spatial_model
        self.sz_model = size_model
        self.co_model = color_model

        self.type = 'SGDMaster'

        self.features = features_ranked
        self.clf_models = {f : SGD() for f in features_ranked}

    def _calc_discrim_score(self, obj, context, feature, value):
        orig_size = context.env_size
        objs, count = context.shared_features(feature, value)

        return count / orig_size, objs

    def produce_input(self, feature, obj, context):
        # TODO: make not stupid
        if feature == "location":
            res, conf = self.sp_model.predict(obj, context)
            conf = conf[0]

        # TODO: uncomment after implementing custom models for color and size
        # elif feature == 'color':
        #     res, conf = self.co_model.predict(obj, context)
        #
        # elif feature == 'size':
        #     res, conf = self.sz_model.predict(obj, context)

        else:
            # one of the purely absolute features
            res = context.get_obj_context_value(obj, feature)
            conf = None

        dscore, objs_remaining = self._calc_discrim_score(obj, context, feature, res)
        return dscore, conf, objs_remaining

    def preprocess(self, corpus_dict):
        # raise NotImplementedError
        Xall = {f : [] for f in self.features}
        Yall = {f : [] for f in self.features}

        for context in corpus_dict:
            context.init_spatial_model(self.sp_model)
            for obj, utt in corpus_dict[context]:
                working_set = context
                features_used = set([t[0] for t in utt])

                for f in self.features:
                    dscore, conf, res_objs = self.produce_input(f, obj, context)
                    if conf:
                        x = [dscore, conf]
                    else:
                        x = [dscore]
                    y = (f in features_used)
                    Xall[f].append(x)
                    Yall[f].append(y)

                    if y:
                        working_set = AdaptiveContext(res_objs)
                        working_set.init_spatial_model(self.sp_model)
        return [Xall], [Yall]

    def train(self, Xall_list, Yall_list):
        Xall = Xall_list[0]
        Yall = Yall_list[0]
        for f in self.features:
            print("Training ", f)
            print(Xall[f])
            print(Yall[f])
            self.clf_models[f].train(Xall[f], Yall[f])

    def _incremental_predict(self, obj, context, features):
        for f in features:
            dscore, conf, objs_remaining = self.produce_input(f, obj, context)

            if conf:
                Y, label = self.clf_models[f].predict([dscore, conf])
            else:
                Y, label = self.clf_models[f].predict([dscore])

            if Y:
                c = AdaptiveContext(objs_remaining)
                c.init_spatial_model(self.sp_model)
                return f, label, c

        return None, None, None

    def predict(self, obj, context):
        features = self.features

        while True:
            feature, label, new_context = self._incremental_predict(obj, context, features)

            if not feature:
                return output

            output += (label + ' ')
            context = new_context

            features = list(set(features) - set(feature))
            if not features:
                return output

    def print_function(self):
        for f in self.clf_models:
            print()
            print('----------------------------------')
            print(f)
            coeff, intercept = self.clf_models[f].get_learned_function()
            print("Coeff: ", coeff)
            print("Intercept: ", intercept)
