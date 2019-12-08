import joblib
from os import path
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

    def save_model(self, file_dest):
        joblib.dump(self.clf, file_dest)
    def save_scaler(self, file_dest):
        joblib.dump(self.sc, file_dest)

    def load_model(self, file_dest):
        self.clf = joblib.load(file_dest)
    def load_scaler(self, file_dest):
        self.sc = joblib.load(file_dest)

class SGDPrimeSelector:
    def __init__(self, features_ordered, spatial_model, size_model, color_model):
        # these all should be pretrained
        self.sp_model = spatial_model
        self.sz_model = size_model
        self.co_model = color_model

        self.type = 'SGDPrime'

        self.features = features_ordered
        self.salience_ranking = features_ordered

        self.clf_models = {f : SGD() for f in features_ordered}

    def _calc_discrim_score(self, obj, context, feature, value):
        orig_size = context.env_size
        objs, count = context.shared_features(feature, value)

        return (orig_size - count) / orig_size, objs

    def produce_input(self, feature, obj, context):
        # TODO: make not stupid
        # if feature == "location":
        # #     res, conf = self.sp_model.predict(obj, context)
        # #     conf = conf[0]

        # TODO: uncomment after implementing custom models for color and size
        # elif feature == 'color':
        #     res, conf = self.co_model.predict(obj, context)
        #
        # elif feature == 'size':
        #     res, conf = self.sz_model.predict(obj, context)

        # else:
            # one of the purely absolute features
        res = context.get_obj_context_value(obj, feature)
        conf = None

        dscore, objs_remaining = self._calc_discrim_score(obj, context, feature, res)
        return dscore, res, conf, objs_remaining

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
                    dscore, label, conf, res_objs = self.produce_input(f, obj, context)
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

        # NEW: add calculation of rank based on intercept
        all_ints = {}
        for f in self.clf_models:
            coef, intercept = self.clf_models[f].get_learned_function()
            all_ints[f] = intercept

        self.salience_ranking.sort(key= lambda x: all_ints[x])

    def _incremental_predict(self, obj, context, features):
        for f in features:
            dscore, label, conf, objs_remaining = self.produce_input(f, obj, context)

            if conf:
                Y = self.clf_models[f].predict([[dscore, conf]])[0]
            else:
                # print(self.clf_models[f].predict([[dscore]]))
                Y = self.clf_models[f].predict([[dscore]])[0]

            if Y:
                c = AdaptiveContext(objs_remaining)
                c.init_spatial_model(self.sp_model)
                return f, label, c

        return None, None, None

    def predict(self, obj, context):
        # NEW
        features = self.features

        # type is added as a given
        type = obj.get_feature_class_value("type")
        output = ''
        context.init_spatial_model(self.sp_model)

        while True:
            feature, label, new_context = self._incremental_predict(obj, context, features)
            # print(feature, label)

            if feature:
                output += (label + ' ')
                context = new_context
                features = list(set(features) - set([feature]))
                # print(features)

            else:
                break
        # print("Finished initial walkthrough with results: ", output)
        if output == "marker red ":
            # import pdb; pdb.set_trace()
            pass

        # add type
        type_context_objs, c = context.shared_features("type", type)
        context = AdaptiveContext(type_context_objs)
        context.init_spatial_model(self.sp_model)

        for f in self.salience_ranking:
            if context.env_size <= 1:
                break

            fval = context.get_obj_context_value(obj, f)
            new_context_objs, count = context.shared_features(f, fval)

            if count < context.env_size:
                output += (fval + ' ')
                context = AdaptiveContext(new_context_objs)
                context.init_spatial_model(self.sp_model)

        output += type
        return output

    def print_function(self):
        for f in self.clf_models:
            print()
            print('----------------------------------')
            print(f)
            coeff, intercept = self.clf_models[f].get_learned_function()
            print("Coeff: ", coeff)
            print("Intercept: ", intercept)

    def save_models(self, dest):
        # rospy.loginfo("Saving models...")
        print("Saving models...")
        for f in self.clf_models:
            file = path.join(dest, ("sgd_" + f + '_clf.pkl'))
            self.clf_models[f].save_model(file)

            scfile = path.join(dest, ("sgd_" + f + "_sclr.pkl"))
            self.clf_models[f].save_scaler(scfile)

    def load_models(self, dest):
        for f in self.clf_models:
            file = path.join(dest, ("sgd_" + f + '_clf.pkl'))
            self.clf_models[f].load_model(file)

            scfile = path.join(dest, ("sgd_" + f + "_sclr.pkl"))
            self.clf_models[f].load_scaler(scfile)
