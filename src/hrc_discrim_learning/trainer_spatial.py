#!/usr/bin/env python
from spatial_learning import ImageLocationLearner, ObjectLocationLearner
from trainer_common import TrainHarness
import rospy

if __name__ == '__main__':
    l1 = ImageLocationLearner()
    # l2 = ObjectLocationLearner()
    all_learners = [l1]

    t = TrainHarness('train_spatial', '/train_spatial_input_provider', all_learners, 'spatial')
    t.run_training()

    # obj1 = Object( {'color': 'black', 'location': (-0.0646604374051094, 0.045339491218328476, 0.0005966073367744684), 'id': 2, 'type': 'tablet', 'orientation': (-0.7058852318614105, 0.04568506114684623, -0.04563287990600581, 0.7053768885492748), 'material': 'plastic', 'size_relative': 'large'} )
    # context = t.context
    # obj1 = context.env[0]
    # print("Prediction for object at location ", obj1.get_feature_class_value('location'), ":")
    # print(l1.predict(obj1, context))
