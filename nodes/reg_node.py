#!/usr/bin/env python3.5
import rospy

from hrc_discrim_learning.srv import SpeechRequest, SpeechRequestResponse
from hrc_discrim_learning.msg import Workspace
from hrc_discrim_learning import re_generator
from hrc_discrim_learning import base_classes

class REGnode:
    def __init__(self):
        # TODO set to False when running
        self.test = True

        rospy.init_node("reg_node")
        self.reg_server = rospy.Service("workspace/reg", SpeechRequest, self.handle_speech_req)
        # self.percept_sub = rospy.Subscriber("workspace/perception", Workspace)

        self.reg = re_generator.REG()
        # TODO uncomment (after training) when running
        # self.reg.load_models()

        self.context = None
        self._bootstrap_workspace()
        rospy.spin()

    def _bootstrap_workspace(self):
        msg = rospy.wait_for_message('/workspace/perception', Workspace)
        self.context = base_classes.process_workspace_from_msg(msg)

    def handle_speech_req(self, req):
        if self.test:
            output = "GOT SRV REQUEST: " + str(req.id)
            print(output)
            return SpeechRequestResponse(output)

        if not self.context:
            rospy.loginfo("Service error: No context info available")
            return None

        # search workspace for matching id
        key = None
        for obj in self.context.env:
            if obj.get_feature_val("id") == req.id:
                key = obj
                break

        if not key:
            rospy.loginfo("Service error: matching obj id not found")

        output = self.reg.generate_output(key, context)
        return SpeechRequestResponse(output)

if __name__ == "__main__":
    REGnode()
