#!/usr/bin/env python
from base import AdaptiveContext
from simple_listener import SimpleListener
from hrc_discrim_learning.srv import TrainInput
import env_perception
import rospy

class TrainHarness:
    def __init__(self, name, srv_name, learners):
        # control msg mappings
        self.NEW_ENV    = 2
        self.CONTINUE   = 1
        self.STOP_TRAIN = 0
        self.ERROR      = -1

        self.listener = SimpleListener()
        self.srv      = srv_name

        self.corpus_dict = {}

        rospy.init_node(name)

        self.context = self.init_new_environment()
        self.all_learners = learners

        self.run_training()

    def receive_train_input(self):
        rospy.wait_for_service(self.srv)
        try:
            input_provider = rospy.ServiceProxy(self.srv, TrainInput)
            resp = input_provider(1)
            return resp

        except rospy.ServiceException:
            print("Service call failed: %s")
            return self.ERROR

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

                ref_obj = None
                for obj in self.context.env:
                    if obj.get_feature_class_value('id') == obj_id:
                        ref_obj = obj

                if not ref_obj:
                    continue

                processed_utt = self.listener.get_named_features_as_tuples(obj_utt)

                self.corpus_dict[self.context].append((ref_obj, processed_utt))

    def init_new_environment(self):
        objs = env_perception.bootstrap_env_info()
        context = AdaptiveContext(objs)
        self.corpus_dict[context] = []
        return context

    def train_all_learners(self):
        # import pdb; pdb.set_trace()
        for learner in self.all_learners:
            X, Y = learner.preprocess(self.corpus_dict)
            learner.train(X, Y)
            learner.print_function()

    def end_train_input(self):
        print("Finished train data collection")

        for context in self.corpus_dict:
            for obj, utt in self.corpus_dict[context]:
                print('(Object(', obj.features, ')', utt, ')')

        self.train_all_learners()
