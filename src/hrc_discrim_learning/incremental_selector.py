from hrc_discrim_learning.base import AdaptiveContext

class IncrementalSelector:
    """

    """
    def __init__(self, feature_rank, sp_model, sz_model, co_model):
        self.features = feature_rank

        self.sp_model = sp_model
        self.sz_model = sz_model
        self.co_model = co_model

        self.type = 'Pure_Incremental'

    def preprocess(self, *args):
        pass

    def train(self, *args):
        pass

    def predict(self, object, context):
        # initialize context with relative models
        context.init_spatial_model(self.sp_model)
        type_matching_objs, count = context.shared_features('type', object.get_feature_class_value('type'))
        context = AdaptiveContext(type_matching_objs)
        context.init_spatial_model(self.sp_model)
        remaining_pool = set(context.env) # list of objs
        output = ''

        for f in self.features:
            v = context.get_obj_context_value(object, f)
            result, new_count = context.shared_features(f, v)

            remaining_pool = remaining_pool.intersection(set(result))
            new_count = len(remaining_pool)

            if new_count < count:
                output += (v + ' ')
            count = new_count

            if new_count == 1:
                break

        # always include category
        v = context.get_obj_context_value(object, 'type')
        output += v

        return output

    def print_function(self):
        print(self.features)
