#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
from find_object_2d.msg import ObjectsStamped
from workspace_speech.msg import Workspace, Object
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import QTransform


class PerceptionTf:
    def __init__(self, id_dict):
        rospy.init_node("workspace/perception_node", anonymous=True)
        self.tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(30), debug=True)

        rospy.Subscriber("/objectsStamped", ObjectsStamped, self.perception_callback)
        self.pub = rospy.Publisher("workspace/perception", Workspace, queue_size=2)
        # self.br = tf2_ros.TransformBroadcaster() # not needed?
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.rate = rospy.Rate(10)

        self.bridge = CvBridge()
        self.transform = QTransform()
        self.dict = id_dict # could load this from file or have it predefined somewhere in this file

        rospy.spin()

    def get_transform(self, obj_name, query_time):
        try:
            trans = self.tfBuffer.lookup_transform("map", obj_name, query_time, rospy.Duration(0.5))
            return trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # print('err')
            return None

    def get_color_average(self, img, corners):
        # img = rospy.wait_for_message(self.image_topic, Image)
        # img_msg = self.bridge.imgmsg_to_cv2(img, encoding="rgb8")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        clr_avg = 0
        pxls = 0

        for xi in range(corners[0].x(), corners[1].x()):
            for yi in range(corners[1].y(), corners[2].y()):
                clr = img[xi,yi][0]
                clr_avg += clr
                pxls += 1

        return round(clr_avg / pxls)

    def get_corners(self, hmat, w, h):
        mat = QTransform(hmat)

        l1 = mat.map(QPointF(0, 0))
        r1 = mat.map(QPointF(w, 0))
        l2 = mat.map(QPointF(0, h))
        r2 = mat.map(QPointF(w, h))

        return [l1, r1, l2, r2]

    def perception_callback(self, msg):
        # extract header information from msg
        time = msg.header.stamp

        output = Workspace()
        output.ObjectArray = []

        img = rospy.wait_for_message("rgb/image_rect_color", Image)
        img_msg = self.bridge.imgmsg_to_cv2(img, encoding="rgb8")

        for i in range(0, len(msg.objects), 12):
            id = msg.objects[i]
            type = self.dict[id]
            w  = msg.objects[i+1]
            h  = msg.objects[i+2]

            # get associated transform
            trans = self.get_transform("object_" + id, time)
            if not trans:
                continue

            new_object = Object()
            new_object.id = id
            new_object.x_dim = w
            new_object.y_dim = h
            new_object.pose = trans

            corners = self.get_corners(msg[i+3:i+13], w, h)
            # TODO: add analysis for color and type
            output.ObjectArray.append(new_object)

        self.pub.publish(output)

if __name__ == "__main__":
    PerceptionTf()
