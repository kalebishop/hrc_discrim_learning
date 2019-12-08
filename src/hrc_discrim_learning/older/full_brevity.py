from itertools import chain, combinations, permutations
import math

from hrc_discrim_learning.base import AdaptiveContext

def combo_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def permute_powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(permutations(s, r) for r in range(len(s)+1))

class FullBrevSelector:
    """

    """
    def __init__(self, features, sp_model, sz_model, co_model, n_scope=False):
        self.features = features

        self.sp_model = sp_model
        self.sz_model = sz_model
        self.co_model = co_model

        self.type = 'Greedy_FB'

        self.narrowing_scope = n_scope

    def train(self, *args):
        pass
    def preprocess(self, *args):
        pass
    def print_function(self):
        pass

    def predict(self, object, context):
        # initialize context with relative models
        context.init_spatial_model(self.sp_model)
        type = object.get_feature_class_value('type')
        type_matching_objs, count = context.shared_features('type', type)
        context = AdaptiveContext(type_matching_objs)
        context.init_spatial_model(self.sp_model)

        obj_pool = set(context.env)

        all_combos = (combo_powerset(self.features) if not self.narrowing_scope else permute_powerset(self.features))
        final_fset = self.features

        for fset in all_combos:
            mini_context = context
            count = context.env_size
            output = ''
            for f in fset:
                val = mini_context.get_obj_context_value(object, f)
                res, count = mini_context.shared_features(f, val)

                if self.narrowing_scope:
                    # update context
                    mini_context = AdaptiveContext(res)
                    mini_context.init_spatial_model(self.sp_model)

                else:
                    obj_pool = obj_pool.intersection(set(res))
                    count = len(obj_pool)

                output += (val + ' ')

            if count <= 1:
                break

        output += type
        return output

class GreedyFBSelector:
    """

    """
    def __init__(self, features, sp_model, sz_model, co_model, n_scope=False):
        self.features = features

        self.sp_model = sp_model
        self.sz_model = sz_model
        self.co_model = co_model

        self.type = 'Greedy_FB'

        self.narrowing_scope = n_scope

    def train(self, *args):
        pass

    def preprocess(self, *args):
        pass

    def predict(self, object, context):
        # initialize context with relative models
        context.init_spatial_model(self.sp_model)
        type_matching_objs, count = context.shared_features('type', object.get_feature_class_value('type'))
        context = AdaptiveContext(type_matching_objs)
        context.init_spatial_model(self.sp_model)

        count = context.env_size
        output = ''
        features = set(self.features)

        # TODO: implement narrowing scope option

        while features and count > 1:
            best_feature = None
            best_score = -math.inf

            for f in features:
                val = context.get_obj_context_value(object, f)
                res, new_count = context.shared_features(f, val)

                score = count - new_count
                if score > best_score:
                    best_fval = (f, val)
                    best_score = score

            count -= best_score
            output += (best_fval[1] + ' ')
            features -= set([best_fval[0]])

        # always include category
        output += object.get_feature_class_value('type')

        return output

    def print_function(self):
        pass


    # def predict(self, object, context):
    #     # initialize context with relative models
    #     context.init_spatial_model(self.sp_model)
    #     count = context.env_size
    #     remaining_pool = set(context.env)
    #
    #     all_combos = powerset(self.features)
    #     feature_set = set([('type', context.get_obj_context_value(object, 'type'))])
    #
    #     for fset in all_combos:
