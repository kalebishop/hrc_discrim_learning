#!/usr/bin/env python
from base import Object, AdaptiveContext
from simple_listener import SimpleListener
from spatial_learning import ImageLocationLearner, ObjectLocationLearner
from hrc_discrim_learning.srv import TrainInput
import env_perception
import rospy
import threading

class TrainSpatialData:
    def __init__(self):
        # control msg mappings
        self.NEW_ENV    = 2
        self.CONTINUE   = 1
        self.STOP_TRAIN = 0
        self.ERROR      = -1

        self.listener = SimpleListener()
        self.img_loc_learner = ImageLocationLearner()
        # self.obj_loc_learner = ObjectLocationLearner()

        # self.mappings = rospy.get_param('/perception/objs')

        self.corpus_dict = {}
        rospy.init_node('train_spatial', anonymous=False)
        self.context = self.init_new_environment()

        self.run_training()

    def run_training(self):
        while not rospy.is_shutdown():
            input = self.receive_train_input()
            if input == self.ERROR:
                break
            elif input.control == self.STOP_TRAIN:
                self.end_train_input()
                break
            elif input.control == self.NEW_ENV:
                self.context = self.init_new_environment()
                self.corpus_dict[self.context] = []

            else:
                obj_id = input.id
                obj_utt = input.utterance

                # match id to obj in env
                print("object id = ", obj_id)
                print("context ids = ", [x.features['id'] for x in self.context.env])

                ref_obj = None
                for obj in self.context.env:
                    if obj.get_feature_class_value('id') == obj_id:
                        ref_obj = obj

                processed_utt = self.listener.get_named_features_as_tuples(obj_utt)

                self.corpus_dict[self.context].append((ref_obj, processed_utt))
                print(self.corpus_dict)

    def init_new_environment(self):
        objs = env_perception.bootstrap_env_info()
        context = AdaptiveContext(objs)
        self.corpus_dict[context] = []
        return context

    def end_train_input(self):
        print("Finishing train data collection")
        # rospy.loginfo("Finished collection of training data")
        X, Y = self.img_loc_learner.preprocess(self.corpus_dict)
        self.img_loc_learner.train(X, Y)
        self.img_loc_learner.print_function()
        # self.obj_loc_learner.train(self.corpus_dict)

    def receive_train_input(self):
        rospy.wait_for_service('train_input_provider')
        try:
            input_provider = rospy.ServiceProxy('train_input_provider', TrainInput)
            resp = input_provider(1)
            return resp

        except rospy.ServiceException:
            print("Service call failed: %s")
            return self.ERROR

if __name__ == '__main__':
    trainer = TrainSpatialData()
