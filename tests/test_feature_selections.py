from hrc_discrim_learning.perceptron_selection import IncrementalFeatureSelector
import json
from hrc_discrim_learning.base import Object, AdaptiveContext
from hrc_discrim_learning.spatial_learning import ImageLocationLearner

def get_predictions():
    with open("/ros/catkin_ws/src/hrc_discrim_learning/train/full_envs.json", 'r') as f:
        all_envs = json.load(f)

    test_env = all_envs["notepads, tablets, pen"]

    env_obj_list = []
    all_objs = test_env
    for obj_id in all_objs:
        features = all_objs[obj_id]
        features['id'] = int(obj_id)
        o = Object(features)
        env_obj_list.append(o)

    c = AdaptiveContext(env_obj_list, "notepads, tablets, pen")

    # spatial_model_dest = rospy.get_param("hrc_discrim_learning/spatial_model")
    loc_model = ImageLocationLearner()
    loc_model.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/spatial")

    learner = IncrementalFeatureSelector(loc_model)
    learner.load_models("/ros/catkin_ws/src/hrc_discrim_learning/model/features")

    for test_o in env_obj_list:
        print(learner.predict(test_o, c))

    print(learner.learner.ppn.coef_)
    print(learner.learner.ppn.intercept_)

if __name__ == '__main__':
    get_predictions()
