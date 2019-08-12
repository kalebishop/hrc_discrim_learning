import unittest
import json
from hrc_discrim_learning.base import Object, AdaptiveContext
from hrc_discrim_learning.spatial_learning import ImageLocationLearner
from hrc_discrim_learning.sgd_learner import IncrementalFeatureSelector

class TestModel:
    def __init__(self):
        pass

    def predict(self, X):
        return 1, 1

class TestSelection(unittest.TestCase):
    def setUp(self):
        loc_model = ImageLocationLearner()
        loc_model.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/spatial")

        features_ordered = ['size_relative', 'color', 'material', 'location']
        self.m = IncrementalFeatureSelector(features_ordered, loc_model, 0, 0)

    def test_predictions(self):
        self.m.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/features")

        with open("/ros/catkin_ws/src/hrc_discrim_learning/train/full_envs.json", 'r') as f:
            all_envs = json.load(f)

        obj_dict = all_envs["differ by 1"]
        obj_list = [Object(obj_dict[id]) for id in obj_dict]
        c = AdaptiveContext(obj_list)

        for o in obj_list:
            print(self.m.predict(o, c))

    # def setUp(self):
    #     self.sp_model, self.co_model, self.sz_model = TestModel()
    #
    #     with open("/ros/catkin_ws/src/hrc_discrim_learning/train/full_envs.json", 'r') as f:
    #         all_envs = json.load(f)
    #
    #     self.corpus_dict = {}
    #
    #     for env in all_envs:
    #         env_obj_list = []
    #         for obj_id in env:
    #             features = env[obj_id]
    #             features['id'] = int(obj_id)
    #             o = Object(features)
    #             env_obj_list.append(o)
    #         c = AdaptiveContext(env_obj_list)
    #
    #
    #
    #     loc_model = ImageLocationLearner()
    #     loc_model.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/spatial")
    #
    #
    #     self.rank = ["size_relative", "color", "type", "location"]
    #     self.model = IncrementalFeatureSelector(self.rank, loc_model, '', '')


if __name__ == '__main__':
    unittest.main()
