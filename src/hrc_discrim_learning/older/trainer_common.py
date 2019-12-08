#!/usr/bin/env python
from hrc_discrim_learning.base import AdaptiveContext
from hrc_discrim_learning.simple_listener import SimpleListener
from hrc_discrim_learning.srv import TrainInput
from hrc_discrim_learning import env_perception
import rospy
import csv
from collections import defaultdict

class TrainHarness:
    def __init__(self, name, srv_name, learners, mode):
        # mode i.e. "spatial" or "features"
        self.mode = mode
        # control msg mappings
        self.NEW_ENV    = 2
        self.CONTINUE   = 1
        self.STOP_TRAIN = 0
        self.ERROR      = -1

        self.listener = SimpleListener()
        self.srv      = srv_name

        # Ros params
        self.save_model     = rospy.get_param("hrc_discrim_learning/save_model")
        self.save_dest      = rospy.get_param("hrc_discrim_learning/save_dest")

        self.train_mode     = rospy.get_param("hrc_discrim_learning/train_mode")
        self.train_data_file= rospy.get_param("hrc_discrim_learning/train_data_file")

        self.save_train_data= rospy.get_param("hrc_discrim_learning/save_train_data")
        self.corpus_file    = rospy.get_param("hrc_discrim_learning/corpus_file")

        rospy.init_node(name)

        self.gen_contexts = env_perception.bootstrap_env_info()
        self.context = self.init_new_environment()
        self.corpus_dict = {self.context: []}
        self.all_learners = learners

        # self.run_training()

    def receive_train_input(self):
        rospy.wait_for_service(self.srv)
        try:
            input_provider = rospy.ServiceProxy(self.srv, TrainInput)
            resp = input_provider(1, self.context.name)
            return resp

        except rospy.ServiceException:
            print("Service call failed: %s")
            return self.ERROR

    def run_training(self):
        while not rospy.is_shutdown():
            input = self.receive_train_input()
            if input == self.ERROR:
                return
            elif input.control == self.STOP_TRAIN:
                self.end_train_input()
                return
            elif input.control == self.NEW_ENV:
                self.context = self.init_new_environment()
                if not self.context:
                    self.end_train_input()
                    return
                else:
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
        context = next(self.gen_contexts)
        return context

    def train_all_learners(self):
        for learner in self.all_learners:
            X = self.Xdict[learner.type] + self.Xfromfile[learner.type]
            Y = self.Ydict[learner.type] + self.Yfromfile[learner.type]

            # print(X)
            # print(Y)

            learner.train(X, Y)
            learner.print_function()

    def end_train_input(self):
        print("Finished train data collection")

        # for context in self.corpus_dict:
        #     for obj, utt in self.corpus_dict[context]:
        #         print('(Object(', obj.features, '),', utt, '),')


        # processed train data from this system
        self.Xdict = {}
        self.Ydict = {}

        # processed train data loaded from csv
        self.Xfromfile = defaultdict(lambda: [])
        self.Yfromfile = defaultdict(lambda: [])

        # preprocessing
        for learner in self.all_learners:
            X, Y = learner.preprocess(self.corpus_dict)

            # store processed data
            self.Xdict[learner.type] = X
            self.Ydict[learner.type] = Y

        # train models
        if self.train_mode == "new":
            self.train_all_learners()

        elif self.train_mode == "aggregate":
            # load in old train data from csv
            rospy.loginfo("Loading train data from file...")
            filename = self.corpus_file

            with open(filename, 'r', newline='') as infile:
                r = csv.reader(infile, delimiter=',')
                for row in r:
                    # read data in format "type" | "X" .... "Xn" | "Y"
                    print(row)
                    y = row.pop()
                    label = row[0]
                    x = tuple([float(z) for z in row[1:]])

                    self.Xfromfile[label].append(x)
                    self.Yfromfile[label].append(y)

            self.train_all_learners()

        # save trained model(s)
        if self.save_model:
            destination = self.save_dest
            for learner in self.all_learners:
                learner.save_models(destination)

        # save new train data to csv
        if self.save_train_data:
            rospy.loginfo("Saving train data...")
            filename = self.corpus_data
            with open(filename, 'a') as outfile:
                w = csv.writer(outfile)

                for type in self.Xdict:
                    X = self.Xdict[type]
                    Y = self.Ydict[type]

                    for x, y in zip(X, Y):
                        w.writerow([type] + list(x) + [y])
