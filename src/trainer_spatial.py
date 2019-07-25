#!/usr/bin/env python
from spatial_learning import ImageLocationLearner, ObjectLocationLearner
from trainer_common import TrainHarness

if __name__ == '__main__':
    l1 = ImageLocationLearner()
    # l2 = ObjectLocationLearner()
    all_learners = [l1]

    t = TrainHarness('train_spatial', '/train_spatial_input_provider', all_learners)
    t.run_training()
