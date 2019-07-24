#!/usr/bin/env python
from base import Object, AdaptiveContext
from simple_listener import SimpleListener
from spatial_learning import ImageLocationLearner, ObjectLocationLearner
from hrc_discrim_learning.msg import TrainInput
import env_perception
import rospy

class TrainSpatialData:
    def __init__(self):
        # control msg mappings
        self.cmd_new_env = "TRAIN NEW ENV"
        self.cmd_stop_train = "TRAIN STOP"

        self.listener = SimpleListener()
        self.img_loc_learner = ImageLocationLearner()
        # self.obj_loc_learner = ObjectLocationLearner()

        # self.mappings = rospy.get_param('/perception/objs')

        self.corpus_dict = {}
        rospy.init_node('train_spatial', anonymous=False)
        rospy.Subscriber('/hrc_discrim_learning/training_input', TrainInput, self.receive_train_input)
        # rospy.Subscriber('/hrc_discrim_learning/training_control', std_msgs/String, self.control)
        self.context = self.init_new_environment()

        rospy.spin()

    def init_new_environment(self):
        # TODO: receive message from system abt whether to init a new env
        objs = env_perception.bootstrap_env_info()
        context = AdaptiveContext(objs)
        self.corpus_dict[context] = []
        return context

    def receive_train_input(self, input):
        # rospy.loginfo("Training callback...", input.utterance.data)
        if rospy.get_param('/hrc_discrim_learning/train_new_env') == 'true':
            self.context = self.init_new_environment()
            rospy.set_param('/hrc_discrim_learning/train_new_env', 'false')

        obj_id = input.id
        obj_utt = str(input.utterance.data)

        if obj_utt == self.cmd_stop_train:
            self.end_train_input()
            return

        print("object id = ", obj_id)
        print("context ids = ", [x.features['id'] for x in self.context.env])

        ref_obj = None
        for obj in self.context.env:
            if obj.get_feature_class_value('id') == obj_id:
                ref_obj = obj

        processed_utt = self.listener.get_named_features_as_tuples(obj_utt)

        self.corpus_dict[self.context].append((ref_obj, processed_utt))
        print(self.corpus_dict)

    def end_train_input(self):
        X, Y = self.img_loc_learner.preprocess(self.corpus_dict)
        self.img_loc_learner.train(X, Y)
        self.img_loc_learner.print_function()
        # self.obj_loc_learner.train(self.corpus_dict)

if __name__ == '__main__':
    trainer = TrainSpatialData()
