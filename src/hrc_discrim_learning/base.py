#!/usr/bin/env python
import numpy as np
import math
import rospy
# from hrc_discrim_learning.msg import ObjArray, ObjPercept
"""
Defines base classes for decision-making about the environment
"""

class Object:
    def __init__(self, feature_dict):
        self.features = feature_dict

    def get_feature_class_value(self, feature):
        return self.features[feature]

class Context:
    """
    Base class for adaptive context analysis for REG
    """
    def __init__(self, objs):
        self.env = objs
        self.env_size = len(objs)

    def shared_features(self, feature, value):
        pass

class AdaptiveContext(Context):
    """

    """
    def __init__(self, objs, name=""):
        super().__init__(objs)
        self.spatial_model = 0
        self.name = name
        # what else?
        self._initialize_workspace_location_info()

    def get_obj_context_value(self, obj, feature):
        if feature == 'location':
            return self.obj_location(obj)[0]
        else:
            return obj.get_feature_class_value(feature)

    def shared_features(self, feature, value):
        if not self.spatial_model:
            print("Error: initialize a spatial model before calling this function")
            return []

        res = []
        c = 0

        for obj in self.env:
            if self.get_obj_context_value(obj, feature) == value:
                res.append(obj)
                c+=1
        return res, c

    def init_spatial_model(self, spatial_model):
        self.spatial_model = spatial_model

    def obj_location(self, obj):
        # print(obj.get_feature_class_value("location"))
        return self.spatial_model.predict(obj, self)

    def _initialize_workspace_location_info(self):
        # should all this be calculated dynamically?
        # calculate centroid (based on x, y)
        sum_x = 0
        sum_y = 0

        # store info on max and min x, y, z(workspace bounding box)
        x_bounds = [math.inf, -math.inf]
        y_bounds = [math.inf, -math.inf]
        z_bounds = [math.inf, -math.inf]

        for o in self.env:
            x, y, z = o.get_feature_class_value("location")
            sum_x += x
            sum_y += y

            x_bounds[0] = min(x_bounds[0], x)
            x_bounds[1] = max(x_bounds[1], x)

            y_bounds[0] = min(y_bounds[0], y)
            y_bounds[1] = max(y_bounds[1], y)

            z_bounds[0] = min(z_bounds[0], z)
            z_bounds[1] = max(z_bounds[1], z)

        self.workspace_centroid = (sum_x / self.env_size, sum_y / self.env_size, 0)

        x_net_max = max(abs(x_bounds[0] - self.workspace_centroid[0]), abs(x_bounds[1] - self.workspace_centroid[0]))
        y_net_max = max(abs(y_bounds[0] - self.workspace_centroid[0]), abs(y_bounds[1] - self.workspace_centroid[0]))
        z_net_max = max(abs(z_bounds[0] - self.workspace_centroid[0]), abs(z_bounds[1] - self.workspace_centroid[0]))

        self.bounds = {'x': x_net_max, 'y': y_net_max, 'z': z_net_max}
        self.max_distance_norm = math.hypot(x_net_max, y_net_max)
