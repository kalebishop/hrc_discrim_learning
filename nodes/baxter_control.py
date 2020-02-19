#!/usr/bin/env python

import math
import struct

import cv2
import numpy as np

import rospy

import utils

from crow.srv import (
    RobotArmController,
    RobotHandController,
    RobotReset,
    ToolUse,
    MoveInitial,
    PickUpTool,
    RobotArmControllerResponse,
    RobotHandControllerResponse,
    RobotResetResponse,
    ToolUseResponse,
    MoveInitialResponse,
    PickUpToolResponse,
    RobotArmControllerRequest,
    RobotHandControllerRequest,
    RobotResetRequest,
    ToolUseRequest,
    MoveInitialRequest,
    PickUpToolRequest
)

from crow.msg import TargetMoving
import baxter_interface

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)

from std_msgs.msg import (
    Header,
    Empty
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest
)

from tool_use import ToolManager


class ArmController():
    def __init__(self):
        self.limb_name = None

        self.limb = None
        self.gripper = None
        self.iksvc = None



    def sleep(self, time, is_move):
        if is_move:
            rospy.sleep(time)
        else: # if we are simulating push, we don't need to sleep.
            pass


    def initialize(self):
        self.limb    = baxter_interface.Limb(self.limb_name)
        self.gripper = baxter_interface.Gripper(self.limb_name)
        # self.gripper.calibrate()

        ns = "ExternalTools/" + self.limb_name + "/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)

        self.reset(None, True)

    def move_to(self, pose, is_move=False):
        if pose.orientation.x == 100. and pose.orientation.y == 100. and pose.orientation.z == 100. and pose.orientation.w == 100.:
            pose.orientation = self.limb.endpoint_pose()["orientation"]

        joints = self.ik_request(pose)

        if is_move:
            return self.set_joint(joints)
        else:
            if joints:
                return True
            else:
                return False

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self.iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        else:
            rospy.logwarn("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def set_joint(self, joint_angles):
        if joint_angles:
            self.limb.move_to_joint_positions(joint_angles)
            return True
        else:
            rospy.logwarn("No Joint Angles provided for move_to_joint_positions. Staying put.")
            return False

    def hands(self, is_open, is_move=False):
        if is_move:
            if is_open:
                self.gripper.open()
            else:
                self.gripper.close()
        # self.sleep(1.0, is_move)
        self.sleep(0.5, is_move)
        return True

    def reset(self, status, is_move=False):
        # status is not used but it is necessary so that ArmControllerManager use the action_wrapper template
        if is_move:
            self.limb.move_to_neutral()
        return True

class LeftArmController(ArmController):
    def __init__(self):
        ArmController.__init__(self)
        self.limb_name = 'left'
        self.initialize()

    # def move_to(self, pose, is_move=False):
    #     h = np.array([[ 0.62378128, -0.11955971, 0.10200936],
    #         [ 0.00637212, 0.84129377, -0.01556247],
    #         [-0.30671601, -0.19758409, 1.        ]])
    #     right_arm_frame = np.array([[pose.position.x, pose.position.y]], dtype='float32')
    #     right_arm_frame = np.array([right_arm_frame])
    #     left_arm_frame = cv2.perspectiveTransform(right_arm_frame, h)
    #     pose.position.x = left_arm_frame[0][0][0]
    #     pose.position.y = right_arm_frame[0][0][1]
    #     return ArmController.move_to(self, pose, is_move)

class RightArmController(ArmController):
    def __init__(self):
        ArmController.__init__(self)
        self.limb_name = 'right'
        self.initialize()

class ArmControllerManager():
    def __init__(self):
        self.left_limb = LeftArmController()
        self.right_limb = RightArmController()
        self.is_moving = False

        # These are needed for observing the push dir of a tool
        self.current_index = -1
        self.obs_push_dir = Point(100,100,100)

    def is_moving(self):
        return self.is_moving

    def action_wrapper(self, side, left, right, left_func, right_func, arg, is_move=False):
        while self.is_moving:
            rospy.sleep(1)

        self.is_moving = True
        result = False

        if side == left:
            result = left_func(arg, is_move)
        elif side == right:
            result = right_func(arg, is_move)

        self.is_moving = False
        return result

    def move_arm(self, data):
        result = self.action_wrapper(data.side, RobotArmControllerRequest.SIDE_LEFT, RobotArmControllerRequest.SIDE_RIGHT, self.left_limb.move_to, self.right_limb.move_to, Pose(position=data.position, orientation=data.orientation), is_move=True)
        return RobotArmControllerResponse(result)

    def hand(self, data):
        result = self.action_wrapper(data.side, RobotHandControllerRequest.SIDE_LEFT, RobotHandControllerRequest.SIDE_RIGHT, self.left_limb.hands, self.right_limb.hands, data.status, is_move=True)
        return RobotHandControllerResponse(result)

    def pick_up_tool(self, data):
        tool_manager = ToolManager(data.type)
        tool = tool_manager.get_tool()

        if data.is_pick_up:
            return self._pick_up_tool(tool)
        else:
            return self._put_down_tool(tool)

    def _pick_up_tool(self, tool):
        limb = self.right_limb

        self.is_moving = True

        result = limb.hands(True, is_move=True)
        rospy.sleep(0.1)
        if result:
            rospy.loginfo("reset hand")
            result = limb.set_joint(tool.pick_up_tool_hover_robot_angles())
            rospy.sleep(0.1)
            if result:
                rospy.loginfo("initial hover")
                result = limb.set_joint(tool.pick_up_tool_robot_angles())
                rospy.sleep(0.1)
                if result:
                    rospy.loginfo("line up")
                    result = limb.hands(False, is_move=True)
                    rospy.sleep(0.1)
                    if result:
                        rospy.loginfo("grasp")
                        result = limb.set_joint(tool.pick_up_tool_hover_robot_angles())
                        rospy.sleep(0.1)
                        if result:
                            rospy.loginfo("pick it up")
                            result = limb.reset(False, is_move=True)
                            if result:
                                rospy.loginfo("reset")

        self.is_moving = False
        return PickUpToolResponse(result)

    def _put_down_tool(self, tool):
        limb = self.right_limb

        self.is_moving = True

        result = limb.reset(False, is_move=True)
        rospy.sleep(0.1)
        if result:
            rospy.loginfo("reset")
            if result:
                result = limb.set_joint(tool.pick_up_tool_hover_robot_angles())
                rospy.sleep(0.1)
                if result:
                    rospy.loginfo("initial hover")
                    result = limb.set_joint(tool.pick_up_tool_robot_angles())
                    rospy.sleep(0.1)
                    if result:
                        rospy.loginfo("put down")
                        result = limb.hands(True, is_move=True)
                        rospy.sleep(0.1)
                        if result:
                            rospy.loginfo("release")
                            result = limb.set_joint(tool.pick_up_tool_hover_robot_angles())
                            rospy.sleep(0.1)
                            if result:
                                rospy.loginfo("hover back")
                                result = limb.reset(False, is_move=True)
                                rospy.sleep(0.1)
                                if result:
                                    rospy.loginfo("reset")


        self.is_moving = False
        return PickUpToolResponse(result)

    def tool_use(self, data):
        tool_manager = ToolManager(data.type)
        tool = tool_manager.get_tool()
        tool.set_usage(data.usage)

        node_name = rospy.get_name()
        use_kb_input  = rospy.get_param(node_name + "/use_kb_input", False)
        reset_arm     = rospy.get_param(node_name + "/reset_arm", False)
        rospy.logwarn("Currently using tool: {}, side: {}, type: {}".format(tool, tool.side, tool.usage))

        ret_obs_push_dir = Point(100,100,100)

        limb = None
        if tool.get_tool_side() == "left":
            limb = self.left_limb
        elif tool.get_tool_side() == "right":
            limb = self.right_limb

        if tool.usage == 'push':
            init_push_height = tool.get_height_to_desk_when_push()
        elif tool.usage =='pull':
            init_push_height = tool.get_height_to_desk_when_pull()

        while self.is_moving:
            rospy.sleep(1)


        end_effector_before_push = Point()
        self.is_moving = True
        is_move = data.is_move_object
        initial_position = tool.get_initial_position(data.init_pos, data.push_direction)
        initial_position.z = tool.get_height_to_desk_initial()
        initial_position.x -= 0.03
        result = limb.move_to(Pose(initial_position, tool.get_quaternion(data.push_direction)), is_move)
        if result:
            rospy.loginfo("initial hover")
            initial_position.z = init_push_height
            result = limb.move_to(Pose(initial_position, tool.get_quaternion(data.push_direction)), is_move)
            if result:
                rospy.loginfo("go down")
                OFFSET = 0.0001
                normalization = math.sqrt(data.push_direction.x ** 2 + data.push_direction.y ** 2 + OFFSET)

                final_position = Point()
                final_position.x = initial_position.x + data.push_direction.x + tool.get_extra_space() * data.push_direction.x / normalization
                final_position.y = initial_position.y + data.push_direction.y + tool.get_extra_space() * data.push_direction.y / normalization
                final_position.z = init_push_height

                mid_push_x = data.push_direction.x * 0.5
                mid_push_y = data.push_direction.y * 0.5
                mid_norm   = math.sqrt(mid_push_x ** 2 + mid_push_y ** 2 + OFFSET)

                mid_position = Point()
                mid_position.x = initial_position.x + mid_push_x + tool.get_extra_space() * mid_push_x / mid_norm
                mid_position.y = initial_position.y + mid_push_y + tool.get_extra_space() * mid_push_y / mid_norm
                mid_position.z = final_position.z

                end_effector_before_push = limb.limb.endpoint_pose()["position"]

                self.obs_push_dir = Point(100,100,100)
                rospy.loginfo("move arm to midpoint of push.")
                result = limb.move_to(Pose(mid_position, tool.get_quaternion(data.push_direction)), is_move)
                if result:
                    rospy.loginfo("moving arm to final position.")
                    result = limb.move_to(Pose(final_position, tool.get_quaternion(data.push_direction)), is_move)
                    if result:
                        limb.sleep(1.0, is_move)
                        ret_obs_push_dir = self.obs_push_dir

                        final_position.z = tool.get_height_to_desk_initial()
                        rospy.loginfo("Lifting arm up...")
                        result = limb.move_to(Pose(final_position, tool.get_quaternion(data.push_direction)), is_move)
                        if result:
                            if use_kb_input and is_move:
                                input("Remove block from velcro if necessary....")
                            rospy.loginfo("hover again")
                            if reset_arm:
                                result = limb.reset(False, is_move)
                            if result:
                                rospy.loginfo("reset")


        self.is_moving = False

        return ToolUseResponse(Point(), Point(), ret_obs_push_dir, end_effector_before_push, result)

    def move_initial(self, data):
        while self.is_moving:
            rospy.sleep(1)

        self.is_moving = True

        is_move = data.is_move_object
        block_height = -0.14
        # add offset due to the sucker is not at the center
        # data.initial_position.x += 0.015
        # data.initial_position.y -= 0.01
        # data.final_position.y -= 0.015

        data.initial_position.z = -0.0 # hover over the object
        quaternion = Quaternion(100., 100., 100., 100.)
        result = self.left_limb.move_to(Pose(data.initial_position, quaternion), is_move)
        if result: # touch object
            # The more to right the arm has to move, the further down it needs to
            # bring the arm to successfully pick up the block it seems.
            if data.initial_position.y > 0.089:
                rospy.loginfo("initial hover")
                block_height = -0.132
            else:
                block_height = -0.14

            data.initial_position.z = block_height
            result = self.left_limb.move_to(Pose(data.initial_position, quaternion), is_move)
            if result: # suck object
                rospy.loginfo("touch object")
                result = self.left_limb.hands(False, is_move)
                if result: # pick it up
                    rospy.loginfo("suck it up")
                    # data.initial_position.z = -0.05
                    data.initial_position.z = 0.05
                    quaternion = Quaternion(100., 100., 100., 100.)
                    result = self.left_limb.move_to(Pose(data.initial_position, quaternion), is_move)
                    if result: # move to final location in the air
                        rospy.loginfo("up in the air")
                        # data.final_position.z = -0.05
                        data.final_position.z = 0.05
                        result = self.left_limb.move_to(Pose(data.final_position, quaternion), is_move)
                        if result: # put object down
                            rospy.loginfo("move to location up in the air")
                            # data.final_position.z = -0.1404921959539664
                            data.final_position.z = block_height
                            result = self.left_limb.move_to(Pose(data.final_position, quaternion), is_move)
                            if result: # release object
                                rospy.loginfo("put it down")
                                result = self.left_limb.hands(True, is_move)
                                if result: # reset arm to neutral
                                    rospy.loginfo("release")
                                    data.final_position.z = -0.1
                                    result = self.left_limb.move_to(Pose(data.final_position, quaternion), is_move)
                                    if result: # release
                                        rospy.loginfo("arm back up in the arm")
                                        result = self.left_limb.reset(False, is_move)
                                        if result:
                                            rospy.loginfo("reset")

        self.is_moving = False

        return MoveInitialResponse(result)

    def reset(self, data, is_move=False):
        result = self.action_wrapper(data.side, RobotResetRequest.SIDE_LEFT,
                                     RobotResetRequest.SIDE_RIGHT, self.left_limb.reset,
                                     self.right_limb.reset, data.status, is_move)
        return RobotResetResponse(result)

    def get_push_dir(self, data):
        """Observes direction of tool push."""
        if data.mode == TargetMoving.MODE_START:
            self.current_index = data.sample_index
            self.init_pos_tool_x, self.init_pos_tool_y = (data.position.tool.x, data.position.tool.y)
            rospy.logwarn("Tool moving (ind: {})!".format(self.current_index))

        elif self.current_index == data.sample_index and data.mode == TargetMoving.MODE_STOP:
            rospy.logwarn("Tool stopped moving (ind: {})!".format(self.current_index))
            final_pos_tool_x, final_pos_tool_y = (data.position.tool.x, data.position.tool.y)

            self.obs_push_dir = Point()
            self.obs_push_dir.x = final_pos_tool_x - self.init_pos_tool_x
            self.obs_push_dir.y = final_pos_tool_y - self.init_pos_tool_y

if __name__ == '__main__':
    try:
        rospy.init_node('robot_arm_controller', anonymous=True)

        manager = ArmControllerManager()
        arm_move = rospy.Service('arm_move', RobotArmController, manager.move_arm)
        hand = rospy.Service('hand_move', RobotHandController, manager.hand)
        tool_use = rospy.Service('tool_use', ToolUse, manager.tool_use)
        move_initial = rospy.Service('move_initial', MoveInitial, manager.move_initial)
        reset = rospy.Service('reset', RobotReset, manager.reset)
        pick_up_tool = rospy.Service('pick_up_tool', PickUpTool, manager.pick_up_tool)
        rospy.Subscriber("/crow/tool_moving_detector", TargetMoving, manager.get_push_dir)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
