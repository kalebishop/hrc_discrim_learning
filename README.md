# HRC discrim_learning
### kayleigh bishop
#### Another repo for the summer-fall 2019 situated REG task
Learning to naturalistically verbally discriminate objects across various environments for human-robot collaboration.

***

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

***
```
roslaunch hrc_discrim_learning training.launch
```
