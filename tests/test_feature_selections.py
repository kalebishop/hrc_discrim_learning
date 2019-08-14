import unittest
import json

from hrc_discrim_learning.base import Object, AdaptiveContext
from hrc_discrim_learning.spatial_learning import ImageLocationLearner

from hrc_discrim_learning.sgd_learner import SGDPrimeSelector
from hrc_discrim_learning.incremental_selector import IncrementalSelector
from hrc_discrim_learning.full_brevity import GreedyFBSelector, FullBrevSelector


# class TestSGDSelection(unittest.TestCase):
#     def setUp(self):
#         loc_model = ImageLocationLearner()
#         loc_model.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/spatial")
#
#         features_ordered = ['size_relative', 'color', 'material', 'location']
#         self.m = SGDPrimeSelector(features_ordered, loc_model, 0, 0)
#
#     def test_predictions(self):
#         self.m.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/features")
#
#         with open("/ros/catkin_ws/src/hrc_discrim_learning/train/full_envs.json", 'r') as f:
#             all_envs = json.load(f)
#
#         obj_dict = all_envs["differ by 1"]
#         obj_list = [Object(obj_dict[id]) for id in obj_dict]
#         c = AdaptiveContext(obj_list)
#
#         for o in obj_list:
#             print(self.m.predict(o, c))

# class TestIncrementalSelection(unittest.TestCase):
#     def setUp(self):
#         loc_model = ImageLocationLearner()
#         loc_model.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/spatial")
#
#         rank = ['color', 'material', 'size_relative', 'location']
#
#         self.m = IncrementalSelector(rank, loc_model, 0, 0)
#
#     def test_predictions(self):
#         with open("/ros/catkin_ws/src/hrc_discrim_learning/train/full_envs.json", 'r') as f:
#             all_envs = json.load(f)
#
#         obj_dict = all_envs["differ by 1"]
#         obj_list = [Object(obj_dict[id]) for id in obj_dict]
#         c = AdaptiveContext(obj_list)
#
#         for o in obj_list:
#             print(self.m.predict(o, c))

# class TestGreedyFBSelection(unittest.TestCase):
#     def setUp(self):
#         loc_model = ImageLocationLearner()
#         loc_model.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/spatial")
#
#         rank = ['color', 'material', 'size_relative', 'location']
#
#         self.m = GreedyFBSelector(rank, loc_model, 0, 0)
#
#     def test_predictions(self):
#         with open("/ros/catkin_ws/src/hrc_discrim_learning/train/full_envs.json", 'r') as f:
#             all_envs = json.load(f)
#
#         obj_dict = all_envs["differ by 1"]
#         obj_list = [Object(obj_dict[id]) for id in obj_dict]
#         c = AdaptiveContext(obj_list)
#
#         for o in obj_list:
#             print(self.m.predict(o, c))

class TestFBSelection(unittest.TestCase):
    def setUp(self):
        loc_model = ImageLocationLearner()
        loc_model.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/spatial")

        rank = ['color', 'material', 'size_relative', 'location']

        self.m = FullBrevSelector(rank, loc_model, 0, 0)

    def test_predictions(self):
        with open("/ros/catkin_ws/src/hrc_discrim_learning/train/full_envs.json", 'r') as f:
            all_envs = json.load(f)

        obj_dict = all_envs["differ by 1"]
        obj_list = [Object(obj_dict[id]) for id in obj_dict]
        c = AdaptiveContext(obj_list)

        for o in obj_list:
            print(self.m.predict(o, c))



if __name__ == '__main__':
    unittest.main()
