#!/usr/bin/env python3
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
import matplotlib.pyplot as plt
import joblib
from os import path

def angle_between(v1, unit_vec):
    # v1 and v2 must have same dimensions
    norm = np.linalg.norm(v1)
    if norm == 0:
        return 3.14
    v1 = v1 / norm
    return np.arccos(np.clip(np.dot(v1, unit_vec), -1.0, 1.0))

class LocationInfoLearner:
    """ Base class for spatial relationship learning.
    """
    def __init__(self):
        self.type = ""

        self.direction_clf = LogisticRegression()
        self.proximity_clf = LogisticRegression()

        self.direction_vocab = None
        self.location_vocab = None

        self.plane_vec = (1, 0, 0)
        self.z_vec = (0, 0, 1)

    def load_models(self, destination_folder):
        direction_path = path.join(destination_folder, self.type + "_direction.pkl")
        self.direction_clf = joblib.load(direction_path)

        # TODO uncomment when running on complete train data
        # proximity_path = path.join(destination_folder, self.type + "_proximity.pkl")
        # self.proximity_clf = joblib.load(proximity_path)

    def save_models(self, destination_folder):
        direction_path = path.join(destination_folder, self.type + "_direction.pkl")
        joblib.dump(self.direction_clf, direction_path)

        # TODO uncomment when running on complete train data
        # proximity_path = path.join(destination_folder, self.type + "_proximity.pkl")
        # joblib.dump(self.proximity_clf, proximity_path)

    def preprocess(self, corpus_data):
        pass

    """ This version of train and predict uses spherical coordinates and a multivariable classifier -
    a better option for when data is plentiful and we want to reduce comp time

    # def train(self, X, Y):
    #     # X is a list of vectors
    #     # Y is a list of the terms used
    #     x = []
    #
    #     for vec, term in zip(X, Y):
    #         theta = angle_between((vec[0], vec[1], 0), self.plane_vec)
    #         phi = angle_between(vec, self.z_vec)
    #         print(theta, phi)
    #         x.append([theta, phi])
    #
    #     self.direction_vocab.fit(x, Y)

    # def predict(self, vec):
    #     theta = angle_between((vec[0], vec[1], 0), self.plane_vec)
    #     phi = angle_between(vec, self.z_vec)
    #     return self.direction_clf.predict([[theta, phi]])
    """

    def train(self, X, Y):
        # X is a list of object vectors
        # Y is a list of the terms used
        dir_x = []
        dir_y = []

        loc_x = []
        loc_y = []

        for vec, term in zip(X, Y):
            if term in self.direction_vocab:
                for direction in self.direction_vocab:
                    dir_vec = self.direction_vocab[direction]

                    theta = angle_between(vec, dir_vec)

                    dir_x.append([theta])
                    dir_y.append((direction==term))

            elif term in self.location_vocab:
                for location in self.location_vocab:
                    d = np.linalg.norm(vec)
                    loc_x.append([d])
                    loc_y.append(term)

        if dir_x:
            self.direction_clf.fit(dir_x, dir_y)
        if loc_x:
            self.proximity_clf.fit(loc_x, loc_y)

    def predict_from_vector(self, vec):
        all_scores = []

        for direction in self.direction_vocab:
            dir_vec = self.direction_vocab[direction]

            theta = angle_between(vec, dir_vec)
            score = self.direction_clf.predict_proba([[theta]])[:,1]
            all_scores.append((direction, score))

        d = np.linalg.norm(vec)
        # for location in self.location_vocab:
        #     score = self.proximity_clf.predict_proba([[theta]])[:,1]
        #     all_scores.append((location, score))

        max_term = None
        max_score = -math.inf
        for term, score in all_scores:
            if score > max_score:
                max_score = score
                max_term = term

        return max_term, max_score

    def print_learned_function(self, key):
        X = []
        for i in np.linspace(0, math.pi, 90):
            X.append([i])

        # X = np.array(X)
        # X.reshape(1, -1)

        Y = self.direction_clf.predict_proba(X)[:,1]

        plt.scatter(X, Y)
        plt.xlabel('theta')
        plt.ylabel(key)

        plt.show()

class ImageLocationLearner(LocationInfoLearner):
    """
    """
    def __init__(self):
        super().__init__()
        self.type = "img"


        self.direction_vocab = {"right": (1, 0), "left": (-1, 0), "front": (0, 1), "back": (0, -1)}
        self.location_vocab = {"center": (0, 0, 0)}

    def preprocess(self, corpus_data):
        X = []
        Y = []


        for context in corpus_data:
            # for debugging
            print("Calculated centroid:", context.workspace_centroid)
            print()
            # # # # #
            for obj, utt in corpus_data[context]:
                for cls, data in utt:
                    if cls == "location":
                        xi, yi, zi = obj.get_feature_class_value("location")
                        obj_vector = (xi, yi)

                        # for debugging
                        # id = obj.get_feature_class_value('id')
                        # print()
                        # print('named:', data)
                        # print("obj", id, "loc", xi, yi, zi, "vec", obj_vector)
                        # # # # #

                        X.append(obj_vector)
                        Y.append(data) # term

        return X, Y

    def predict(self, obj, context):
        location = obj.get_feature_class_value("location")
        x, y, z = np.subtract(location, context.workspace_centroid)
        obj_workspace_vector = (x, y)
        # print("in spatial_learning:", obj_workspace_vector)
        return self.predict_from_vector(obj_workspace_vector)

    def print_function(self):
        self.print_learned_function('image_based')

class ObjectLocationLearner(LocationInfoLearner):
    """
    """
    def __init__(self):
        super().__init__()
        self.type = 'obj'

        self.direction_vocab = {"right": (1, 0, 0), "left": (-1, 0, 0), "front": (0, -1, 0), "back": (0, 1, 0), "above": (0, 0, 1), "below": (0, 0, -1)}
        self.location_vocab = {"near": (0, 0, 0), "next": (0, 0, 0)}

        self.dimensions = ['x', 'y', 'z']

    def _calculate_object_bounding_box(self, object):
        # object must have a "dimensions" feature
        x_dim, y_dim, z_dim = object.get_feature_class_value("dimensions")
        # assuming for now that the location data of the object refers to its approx centroid - might not stick with that
        x, y, z = object.get_feature_class_value("location")

        x_min = x - x_dim / 2
        x_max = x + x_dim / 2

        y_min = y - y_dim / 2
        y_max = y + y_dim / 2

        z_min = z - z_dim / 2
        z_max = z + z_dim / 2

        # bounding_box = {"x": (x_min, x_max), "y": (y_min, y_max), "z": (z_min, z_max)}
        bounding_box = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        return bounding_box

    def _calculate_object_vector(self, obj1, obj2):
        vec = [0, 0, 0]
        box = self._calculate_object_bounding_box(obj1)
        coords = obj2.get_feature_class_value("location")

        for i in range(len(coords)):
            min_bound, max_bound = box[i]
            p = coords[i]

            if abs(min_bound - p) > abs(max_bound - p):
                closest_bound = min_bound
            else:
                closest_bound = max_bound
            component = p - closest_bound
            vec[i] = component

        vec = tuple(vec)
        return vec

    def preprocess(self, corpus_data):
        X = []
        Y = []

        for context in corpus_data:
            for obj, utt in corpus_data[context]:
                for cls, data in utt:
                    if cls == 'object_location':
                        term = data[0]
                        compare_obj = data[1]

                        vec = self._calculate_object_vector(compare_obj, obj)
                        X.append(vec)
                        Y.append(term)
        return X, Y

    def predict(self, obj1, obj2):
        obj_vector = self._calculate_object_vector(obj1, obj2)
        return self.predict_from_vector(obj_vector)

    def print_function(self):
        self.print_learned_function('object-based')
