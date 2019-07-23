#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from aruco_msgs.msg import MarkerArray

def callback(data):
    print('1', data)

def listener():
    rospy.init_node('listener', anonymous=False)
    rospy.Subscriber('/my_aruco/markers', MarkerArray, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
