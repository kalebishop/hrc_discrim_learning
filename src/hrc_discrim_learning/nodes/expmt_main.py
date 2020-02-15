import rospy

from svox_tts.srv import Speech
from hrc_discrim_learning.srv import SpeechRequest
from hrc_discrim_learning.msg import Workspace

class Main:
    # TODO finish
    def __init__(self):
        rospy.init_node('workspace/experiment')

        rospy.wait_for_service('workspace/reg')
        self.reg_srv = rospy.ServiceProxy('workspace/reg', SpeechRequest)

        rospy.wait_for_service('svox_tts/speech')
        self.tts_srv = rospy.ServiceProxy('svox_tts/speech', Speech)

        self.percept_sub  = rospy.Subscriber('workspace/perception', Workspace)

    def get_output_string(self, object_id):
        return self.reg_srv(object_id)

    def speech_pub(self, string):
        self.tts_srv(string)
