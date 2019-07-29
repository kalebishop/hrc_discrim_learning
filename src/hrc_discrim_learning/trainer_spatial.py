#!/usr/bin/env python
from hrc_discrim_learning.spatial_learning import ImageLocationLearner, ObjectLocationLearner
from hrc_discrim_learning.trainer_common import TrainHarness
import rospy

if __name__ == '__main__':
    l1 = ImageLocationLearner()
    # l2 = ObjectLocationLearner()
    all_learners = [l1]

    t = TrainHarness('train_spatial', '/train_input_provider', all_learners, 'spatial')
    t.run_training()
