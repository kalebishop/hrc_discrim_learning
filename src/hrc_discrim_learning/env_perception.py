#!/usr/bin/env python
import rospy
from aruco_msgs.msg import MarkerArray
from base import Object
import json

def bootstrap_from_file(mappings):
    filename = rospy.get_param('/hrc_discrim_learning/obj_location_file')
    with open(filename, 'r') as f:
        obj_dict = json.load(f)

    objs = []

    for context in obj_dict:
        for id in obj_dict[context]:
            features = mappings[id]
            features['id'] = int(id)
            features['orientation'] = tuple(obj_dict[context][id]['orientation'])
            features['location'] = tuple(obj_dict[context][id]['location'])

            o = Object(features)
            objs.append(o)

    return objs

def bootstrap_aruco_env_info(mappings):
    rospy.loginfo("Waiting for aruco env bootstrap info...")
    # msg = rospy.wait_for_message('/aruco_pub/markers', MarkerArray)
    key = input("Press enter to capture environment.\n")
    msg = rospy.wait_for_message('/aruco_pub/markers', MarkerArray)
    rospy.loginfo("Bootstrap info received")

    objs = []

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

def bootstrap_env_info():
    with open(rospy.get_param("hrc_discrim_learning/obj_mapping_file")) as f:
        mappings = json.load(f)

    if rospy.get_param('hrc_discrim_learning/use_aruco'):
        return bootstrap_aruco_env_info(mappings)
    else:
        return bootstrap_from_file(mappings)

if __name__ == '__main__':
    bootstrap_env_info()
