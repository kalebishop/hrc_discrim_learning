#!/usr/bin/env python
from hrc_discrim_learning.srv import *
import rospy

class ScriptedTrainer:
    def __init__(self):
        self.training_script = [
            (4, "left block"),
            (1, "right block"),
            (2, "close block"),
            (5, "far block")
        ]

        self.NEW_ENV    = 2
        self.CONTINUE   = 1
        self.STOP_TRAIN = 0
        self.ERROR      = -1

        self.run        = True

    def gen_input(self, req):
        rospy.loginfo("Got request")
        if not self.training_script:
            self.run = False
            return TrainInputResponse(self.STOP_TRAIN, 0, "")
        else:
            id, utt = self.training_script.pop()
            return TrainInputResponse(self.CONTINUE, id, utt)

    def run_train_server(self):
        rospy.init_node('scripted_training')
        s = rospy.Service('train_input_provider', TrainInput, self.gen_input)
        while self.run:
            rospy.sleep(10)

if __name__ == '__main__':
    t = ScriptedTrainer()
    t.run_train_server()

# TRAINING_DATA = [
#     (4, "left block"),
#     (1, "right block"),
#     (2, "close block"),
#     (5, "far block"),
#     (0, "TRAIN STOP")
# ]
#
# def main():
#     rospy.init_node('auto_train')
#     pub = rospy.Publisher('/hrc_discrim_learning/training_input', TrainInput, queue_size=1)
#
#     # rospy.sleep(10)
#
#     i = 0
#     while (not rospy.is_shutdown()) and (i < len(TRAINING_DATA)):
#         rospy.loginfo('Looping through training data')
#         id, utt = TRAINING_DATA[i]
#         pub.publish(TrainInput(id, String(utt)))
#         rospy.sleep(1)
#         i += 1
#
# if __name__ == '__main__':
#     main()
