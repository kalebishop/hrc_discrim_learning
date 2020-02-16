#!/usr/bin/env python
import rospy

from svox_tts.srv import Speech
from hrc_discrim_learning.srv import SpeechRequest
from hrc_discrim_learning.msg import Workspace

class Main:
    # TODO finish
    def __init__(self):
        rospy.init_node('experiment_node')

        rospy.wait_for_service('workspace/reg')
        self.reg_srv = rospy.ServiceProxy('workspace/reg', SpeechRequest)

        # TODO uncomment when running
        # rospy.wait_for_service('svox_tts/speech')
        # self.tts_srv = rospy.ServiceProxy('svox_tts/speech', Speech)

        self.all_obj_ids = []

        # bootsrap initial workpsace data
        self._update_workspace()

        # TODO comment when running
        self.get_output_string(1)

        # TODO finish
        rospy.spin()

    def _update_workspace(self):
        msg = rospy.wait_for_message('workspace/perception', Workspace)

        self.all_obj_ids = []
        for object in msg.ObjectArray:
            self.all_obj_ids.append(object.id)

    def get_output_string(self, object_id):
        return self.reg_srv(object_id)

    def speech_pub(self, string):
        self.tts_srv(string)

if __name__ == "__main__":
    Main()
