#!/usr/bin/env python3
import rospy
import os
import re

from workspace_speech.srv import SpeechRequest
from workspace_speech.msg import Workspace
from workspace_speech.base_classes import * # Object and Context

class Talker:
    def __init__(self):
        self.models = {}

        model_location = rospy.get_param("~speech_model_location", "/home/kaleb/ros/catkin_ws/src/workspace_speech/models/speech")
        feature_speech_models = os.listdir(model_location)
        self.initialize_models(feature_speech_models)

        self.using_features = ["dimension", "location", "color", "size"]

        rospy.init_node("workspace_speech_server", anonymous=True)

        self.context = None

        perception_sub = rospy.Subscriber("/workspace", Workspace, self.perceptual_update)
        s = rospy.Service("workspace_speech", SpeechRequest, self.speech_request)

        rospy.spin()

    def initialize_models(self, saved_models):
        for f in self.using_features:
            self.models[f] = SpeechLearner(f)

            r = re.compile("svm_" + f)
            k = None
            for file in saved_models:
                k = r.search(file)
                if k:
                    break

            self.models[f].load_model(file)

    def perceptual_update(self, msg):
        workspace = process_workspace_from_msg(msg)
        self.context = Context(workspace)

    def speech_request(self, req):
        # server that receives request in the form of an id - fills it in with the description
        id = req.id # id of the object to gen a description for

        # generate description
        # TODO implement!
        return SpeechRequestResponse("")
