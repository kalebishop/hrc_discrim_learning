import rospy
from hrc_discrim_learning.srv import *

rospy.wait_for_service('/train_input_provider')
input_provider = rospy.ServiceProxy('/train_input_provider', TrainInput)

for i in range(5):
    resp = input_provider(1, "notepads, tablet, pen")
    print(resp)

resp = input_provider(0, "")
