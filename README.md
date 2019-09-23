# HRC discrim_learning
### kayleigh bishop
#### Another repo for the summer-fall 2019 situated REG task
Learning to naturalistically verbally discriminate objects across various environments for human-robot collaboration.

***
## Motivation
Referring expression generation (REG) is a classic task in natural language generation with a number of useful applications in robotics. Many canonical algorithms have been proposed in the past decades to solve the problem of how best to refer to objects in the environment. However, these algorithms have been criticized for either failing to capture important aspects of human REs (particularly overspecification) or failing to consistently produce adequately specifying REs (Koolen et al., 2011).

For the task of human-robot collobaration, an important goal is to design a natual language system that can produce descriptions that are easy to understand for human listeners. The most obvious place to look for inspiration is how humans themselves produce REs from their observations of the envirnoment. I propose and here implement an algorithm which draws on both classical and contemporary psychophysics research, particularly from Fukumura's 2018 paper in which she proposes the PASS model of adjective selection and ordering.

REG systems intended for real-life task application require special considerations for the kinds of perceptual information available to the system and the speed of processing. The proposed implementation of the algorithm takes these considerations into account and is designed to work with a real-time perceptual system on a Kinect v2 setup.

## The Task
To evaluate the algorithm and its implementation, we have selected two evaluative tasks: an online interactive survey, and an in-person collaborative task involving a human and a robot equipped with a basic perceptual and speech system. In the online task we aim to collect both ground truth data on human REs as well as human rating of REs produced by the algorithm in comparison to those produced by canonical algorithms. In the interactive HRC task, we aim to evaluate the feasibility of the algorithm in real-life, real-time perceptual situations as well as the appropriateness of the generated REs for human-robot collaboration tasks.

## The Algorithm
Usage of a particular adjective in a definite description is taken to be a linear combination of the uniqueness of the descriptor (i.e. how many objects it eliminates), and the linguistic-perceptual salience of the feature. Because the uniqueness of a descriptor is observable during training, as is information of the ordering constraints of the language, the salience is taken to represent both a weighting of uniqueness and a "bias" towards use of a particular feature. As such, the algorithm implements SGD linear classifiers for each feature which are each trained during any given training session. The classifier then learns a linear threshold for inclusion of the feature in the description for the given object in the particular context.

To produce output, the ordering constrants of the langauge are used as a preference list. For each feature, the best descriptor term is calculated (e.g. "red" for color, "far" for location) as is the uniqueness of that descriptor. This term is then used as input to the learned classification function for the given feature - if the score exceeds the threshold, the adjective is included in the final description. The scope of future features is narrowed, the feature used is eliminated from the list, and the iteration restarts until all features have been checked and a fully specifying description has been produced.

## Main src files:

**base.py**: Defines the classes for Object, Context (i.e. a set of Objects), and AdaptiveContext (which includes extra methods for calculating workspace dimensions, etc.)  
**env_perception.py**: Provides one-time subscription interface with aruco_ros via the bootstrap_env_info() function.  
**simple_listener.py**: Defines SimpleListener, which works as a basic limited-vocab interpreter for natural language input.   
**spatial_learning.py**: Defines the ImageLocationLearner and ObjectLocationLearner classes, used for training a logistic regression model for generating spatial relations w.r.t. the entire workspace and other objects, respectively.  
**trainer_common**: Base training harness used by trainer_spatial and trainer_features.


## Nodes source files:

**trainer_spatial.py**: Instantiates and runs training routine for learning spatial relationships using service call to `/train_spatial_input_provider`.  
**auto_train.py**: Script to automatically return service calls to `/train_spatial_input_provider` with preloaded training data. Only runs if the `use_auto_train` arg is set to `True` in `training.launch`.

## Other files of note:

**basic_aruco.launch**: Launches aruco node and simple camera/display interface. Useful for checking camera and display is working properly.  
**cam_info.yaml**: Defines the camera_info published to `usb_cam/camera_info` on launch.  
**training.launch**: Launchfile for current training system.  

## Dependencies:
usb_cam
aruco_ros
image_pipeline

***
```
roslaunch hrc_discrim_learning spatial_training.launch
roslaunch hrc_discrim_learning feature_training.launch
```
