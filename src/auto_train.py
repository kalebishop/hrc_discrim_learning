#!/usr/bin/env python
from hrc_discrim_learning.srv import *
import rospy

class ScriptedTrainer:
    def __init__(self):
        self.training_script = [
            [
                (4, "left block"),
                (1, "right block"),
                (2, "front block"),
                (5, "back block")
            ],
            # [
            #     (2, "right block"),
            #     (4, "right block"),
            #     (5, "left block"),
            #     (1, "left block")
            # ]
        ]

        self.current_context = self.training_script.pop()

        self.NEW_ENV    = 2
        self.CONTINUE   = 1
        self.STOP_TRAIN = 0
        # self.ERROR      = -1

        self.run        = True

    def gen_input(self, req):
        rospy.loginfo("Got request")
        if not self.current_context:
            if not self.training_script:
                self.run = False
                return TrainInputResponse(self.STOP_TRAIN, 0, "")
            else:
                self.current_context = self.training_script.pop()
                return TrainInputResponse(self.NEW_ENV, 0, "")
        else:
            id, utt = self.current_context.pop()
            return TrainInputResponse(self.CONTINUE, id, utt)

    def run_train_server(self):
        rospy.init_node('scripted_training')
        s = rospy.Service('train_spatial_input_provider', TrainInput, self.gen_input)
        while self.run:
            rospy.sleep(5)

if __name__ == '__main__':
    t = ScriptedTrainer()
    t.run_train_server()
