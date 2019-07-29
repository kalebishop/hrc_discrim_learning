#!/usr/bin/env python
import rospy
from hrc_discrim_learning.trainer_common import TrainHarness
from hrc_discrim_learning.feature_learning import IncrementalLearner
from hrc_discrim_learning.spatial_learning import ImageLocationLearner

if __name__ == "__main__":
    spatial_model_dest = rospy.get_param("hrc_discrim_learning/spatial_model")
    loc_model = ImageLocationLearner()
    loc_model.load_models(spatial_model_dest)

    feat_learner = IncrementalLearner(loc_model)

    all_learners = [feat_learner]

    t = TrainHarness('train_feature', '/train_input_provider', all_learners, 'feature')
    t.run_training()
