#!/usr/bin/env python
import rospy
# from aruco_msgs.msg import MarkerArray
from hrc_discrim_learning.base import Object, AdaptiveContext
import json

def bootstrap_env_info():
    # if rospy.get_param('hrc_discrim_learning/use_perception'):
    if False:
        # TODO: implement
        pass

    else:
        filename = rospy.get_param('/hrc_discrim_learning/env_file')
        # filename = "/ros/catkin_ws/src/hrc_discrim_learning/train/full_envs.json"

        with open(filename, 'r') as f:
            all_contexts = json.load(f)

        for context_name in all_contexts:
            env_obj_list = []
            all_objs = all_contexts[context_name]
            for obj_id in all_objs:
                features = all_objs[obj_id]
                features['id'] = int(obj_id)
                o = Object(features)
                env_obj_list.append(o)

            yield AdaptiveContext(env_obj_list, context_name)

        while True:
            yield None


if __name__ == '__main__':
    bootstrap_env_info()
