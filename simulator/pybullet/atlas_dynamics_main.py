import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import shutil

import cv2
import pybullet as p
import numpy as np

np.set_printoptions(precision=2)

from config.atlas_config import SimConfig
from pnc.atlas_pnc.atlas_interface import AtlasInterface
from util import pybullet_util
from util import util
from util import liegroup

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
args = parser.parse_args()

file = args.file


def set_initial_config(robot, joint_id):
    # shoulder_x
    p.resetJointState(robot, joint_id["l_arm_shx"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_arm_shx"], np.pi / 4, 0.)
    # elbow_y
    p.resetJointState(robot, joint_id["l_arm_ely"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_arm_ely"], np.pi / 2, 0.)
    # elbow_x
    p.resetJointState(robot, joint_id["l_arm_elx"], -np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_arm_elx"], -np.pi / 2, 0.)
    # hip_y
    p.resetJointState(robot, joint_id["l_leg_hpy"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_hpy"], -np.pi / 4, 0.)
    # knee
    p.resetJointState(robot, joint_id["l_leg_kny"], np.pi / 2, 0.)
    p.resetJointState(robot, joint_id["r_leg_kny"], np.pi / 2, 0.)
    # ankle
    p.resetJointState(robot, joint_id["l_leg_aky"], -np.pi / 4, 0.)
    p.resetJointState(robot, joint_id["r_leg_aky"], -np.pi / 4, 0.)


def signal_handler(signal, frame):
    if SimConfig.VIDEO_RECORD:
        pybullet_util.make_video(video_dir, False)
    p.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)

    # p.resetDebugVisualizerCamera(cameraDistance=1.5,
    # cameraYaw=120,
    # cameraPitch=-30,
    # cameraTargetPosition=[1., 0.5, 1.5])

    p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                 cameraYaw=60,
                                 cameraPitch=-20,
                                 cameraTargetPosition=[0., 0., 1.])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=SimConfig.CONTROLLER_DT,
                                numSubSteps=SimConfig.N_SUBSTEP)
    if SimConfig.VIDEO_RECORD:
        video_dir = 'video/atlas_pnc'
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir)

    # Create Robot, Ground
    BASE_JOINT_POS = SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    if file == "data/atlas_block.yaml":
        BASE_JOINT_POS = [0., 0., 1.8 - 0.761]
        block = p.loadURDF(cwd + "/robot_model/ground/block.urdf",
                           [0, 0, 0.15],
                           useFixedBase=True)
    elif file == "data/atlas_stair.yaml":
        stair = p.loadURDF(cwd + "/robot_model/ground/stair.urdf",
                           [0.20, 0, 0.],
                           useFixedBase=True)
    elif file == "data/atlas_slope.yaml":
        gap = p.loadURDF(cwd + "/robot_model/ground/slope.urdf",
                         [0.325, 0, -0.125],
                         useFixedBase=True)
    elif file == 'data/atlas_chimney.yaml':
        lr_chimney = p.loadURDF(cwd + "/robot_model/ground/chimney.urdf",
                                [0, 0, 0],
                                useFixedBase=True)
    elif file == 'data/atlas_lr_chimney_jump.yaml':
        lr_chimney = p.loadURDF(cwd + "/robot_model/ground/lr_chimney.urdf",
                                [0, 0, 0],
                                useFixedBase=True)

    if (file != 'data/atlas_lr_chimney_jump.yaml') and (
            file != 'data/atlas_chimney.yaml'):
        p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])

    robot = p.loadURDF(cwd + "/robot_model/atlas/atlas.urdf", BASE_JOINT_POS,
                       SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0)

    # Construct Interface
    interface = AtlasInterface()

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0

    while (1):
        # Get SensorData
        if SimConfig.SIMULATE_CAMERA and count % (
                SimConfig.CAMERA_DT / SimConfig.CONTROLLER_DT) == 0:
            camera_img = pybullet_util.get_camera_image_from_link(
                robot, link_id['head'], 50, 10, 60., 0.1, 10)
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        rf_height = pybullet_util.get_link_iso(robot, link_id['r_sole'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot, link_id['l_sole'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

        # Get Keyboard Event
        keys = p.getKeyboardEvents()
        if pybullet_util.is_key_triggered(keys, '8'):
            interface.interrupt_logic.b_interrupt_button_eight = True
        elif pybullet_util.is_key_triggered(keys, '5'):
            interface.interrupt_logic.b_interrupt_button_five = True
        elif pybullet_util.is_key_triggered(keys, '4'):
            interface.interrupt_logic.b_interrupt_button_four = True
        elif pybullet_util.is_key_triggered(keys, '2'):
            interface.interrupt_logic.b_interrupt_button_two = True
        elif pybullet_util.is_key_triggered(keys, '6'):
            interface.interrupt_logic.b_interrupt_button_six = True
        elif pybullet_util.is_key_triggered(keys, '7'):
            interface.interrupt_logic.b_interrupt_button_seven = True
        elif pybullet_util.is_key_triggered(keys, '9'):
            interface.interrupt_logic.b_interrupt_button_nine = True
        elif pybullet_util.is_key_triggered(keys, '1'):
            interface.interrupt_logic.b_interrupt_button_one = True
        elif pybullet_util.is_key_triggered(keys, '3'):
            interface.interrupt_logic.b_interrupt_button_three = True
        # Compute Command
        if SimConfig.PRINT_TIME:
            start_time = time.time()
        command = interface.get_command(copy.deepcopy(sensor_data))

        if SimConfig.PRINT_TIME:
            end_time = time.time()
            print("ctrl computation time: ", end_time - start_time)

        # Apply Trq
        pybullet_util.set_motor_trq(robot, joint_id, command['joint_trq'])

        # Save Image
        if (SimConfig.VIDEO_RECORD) and (count % SimConfig.RECORD_FREQ == 0):
            # frame = pybullet_util.get_camera_image([1.2, 0.5, 1.], 2.0, 120,
            # -15, 0, 60., 1920, 1080,
            # 0.1, 100.)
            frame = pybullet_util.get_camera_image([0.5, 0.5, 1.], 2.0, 210,
                                                   -10, 0, 60., 1920, 1080,
                                                   0.1, 100.)
            frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
            filename = video_dir + '/step%06d.jpg' % count
            cv2.imwrite(filename, frame)

        p.stepSimulation()

        # time.sleep(dt)
        t += dt
        count += 1
