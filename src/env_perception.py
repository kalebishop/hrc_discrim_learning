#!/usr/bin/env python
import rospy
from aruco_msgs.msg import MarkerArray
# from collections import OrderedDict
# from geometry_msgs.msg import PoseWithCovariance
# from hrc_discrim_learning.msg import ObjPercept, ObjArray
from base import Object

def bootstrap_env_info():
    rospy.loginfo("Waiting for aruco env bootstrap info...")
    msg = rospy.wait_for_message('/aruco_pub/markers', MarkerArray)
    rospy.loginfo("Bootstrap info received")

    objs = []

    mappings = rospy.get_param('/perception/objs')

    for marker in msg.markers:
        id = str(marker.id)
        features = mappings[id]
        pt = marker.pose.pose.position
        ot = marker.pose.pose.orientation
        features['location']    = (pt.x, pt.y, pt.z) # Point obj
        features['orientation'] = (ot.x, ot.y, ot.z, ot.w) #  Quaternion obj
        features['id'] = marker.id

        o = Object(features)
        objs.append(o)

    return objs

if __name__ == '__main__':
    bootstrap_env_info()

# class ObjectPerception:
#     def __init__(self):
#         rospy.init_node('env_perception', anonymous=False)
#         r = rospy.Rate(1)
#
#         item_ids = rospy.get_param('/perception/objs')
#         self.env_dict = {}
#         for name, id in item_ids.items():
#             d = OrderedDict()
#             d['id'] = id
#             d['name'] = name
#             d['pose'] = None
#             d['seen'] = False
#
#             self.env_dict[id] = d
#
#         rospy.Subscriber('/my_aruco/markers', MarkerArray, self.process_env)
#         self.pub = rospy.Publisher('/hrc_discrim_learning/perception', ObjArray, queue_size=2)
#
#         rospy.spin()
#
#     def process_env(self, marker_array):
#         output = []
#         current_time = rospy.get_time()
#         for marker in marker_array.markers:
#             self.env_dict[marker.id]["pose"] = marker.pose.pose
#             self.env_dict[marker.id]["seen"] = current_time
#             output.append(ObjPercept(*self.env_dict[marker.id].values()))
#
#         self.pub.publish(output)
#         # rospy.sleep(5)
#
# if __name__ == '__main__':
#     ObjectPerception()
