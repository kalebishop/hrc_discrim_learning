#!/usr/bin/env python
from hrc_discrim_learning.msg import TrainInput
from std_msgs.msg import String
import rospy

TRAINING_DATA = [
    (4, "left block"),
    (1, "right block"),
    (2, "close block"),
    (5, "far block"),
    (0, "TRAIN STOP")
]

def main():
    rospy.init_node('auto_train')
    pub = rospy.Publisher('/hrc_discrim_learning/training_input', TrainInput, queue_size=1)

    i = 0
    while (not rospy.is_shutdown()) and (i < len(TRAINING_DATA)):
        rospy.loginfo('Looping through training data')
        id, utt = TRAINING_DATA[i]
        pub.publish(TrainInput(id, String(utt)))
        rospy.sleep(1)
        i += 1

if __name__ == '__main__':
    main()
