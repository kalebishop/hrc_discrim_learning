#!/usr/bin/env python
from hrc_discrim_learning.srv import *
import rospy
import json

class ScriptedTrainer:
    def __init__(self):
        # TODO: fix
        ##########################
        with open(rospy.get_param("hrc_discrim_learning/train_data_file")) as f:
            d = json.load(f)

        self.training_script = []

        for context in d:
            for id, utt in d[context].items():
                id = int(id)
                self.training_script.append((id, utt))

        self.training_script = [self.training_script]
        ########################

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
        s = rospy.Service('train_input_provider', TrainInput, self.gen_input)
        while self.run and not rospy.is_shutdown():
            rospy.sleep(3)

if __name__ == '__main__':
    t = ScriptedTrainer()
    t.run_train_server()
