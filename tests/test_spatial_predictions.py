#!/usr/bin/env python
import unittest
import rospy
import json

from hrc_discrim_learning.base import AdaptiveContext, Object
from hrc_discrim_learning.spatial_learning import ImageLocationLearner

class TestSpatialPrediction(unittest.TestCase):
    def setUp(self):
        rospy.loginfo("Initializing test for spatial predictions")
        self.model = ImageLocationLearner()

        self.model.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/spatial_regressors")

    def test_prediction(self):
        features = {
            "location": [1, .2, .2],
            "orientation": [0, 0, 0, 0],
            "description" : "right"
          }


        obj_file = "/ros/catkin_ws/src/hrc_discrim_learning/train/objects.json"
        with open(obj_file, 'r') as f:
            obj_dict = json.load(f)

        all_objs = [Object(x) for id, x in obj_dict.items()]
        context = AdaptiveContext(all_objs)

        for o in all_objs:
            self.assertEqual(o.get_feature_class_value('description'),
                self.model.predict(o, context))

if __name__ == '__main__':
    unittest.main()
