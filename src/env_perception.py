#!/usr/bin/env python
import rospy
from aruco_msgs.msg import MarkerArray
# from collections import OrderedDict
# from geometry_msgs.msg import PoseWithCovariance
# from hrc_discrim_learning.msg import ObjPercept, ObjArray
from base import Object

def bootstrap_env_info():
    rospy.loginfo("Waiting for aruco env bootstrap info...")
    # msg = rospy.wait_for_message('/aruco_pub/markers', MarkerArray)
    key = input("Press enter to capture environment.\n")
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
