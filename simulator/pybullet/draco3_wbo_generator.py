import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import time
from datetime import datetime

import pybullet as pb
import numpy as np

np.set_printoptions(precision=4)
import ipdb

from config.draco3_wbo_config import SimConfig
from util import pybullet_util
from util import util
from collections import OrderedDict

from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem

import copy

import matplotlib.pyplot as plt

from casadi import *


def set_initial_config(robot, joint_id):
    # Upperbody
    pb.resetJointState(robot, joint_id["l_shoulder_aa"], np.pi / 6, 0.)
    pb.resetJointState(robot, joint_id["l_elbow_fe"], -np.pi / 2, 0.)
    pb.resetJointState(robot, joint_id["r_shoulder_aa"], -np.pi / 6, 0.)
    pb.resetJointState(robot, joint_id["r_elbow_fe"], -np.pi / 2, 0.)

    # Lowerbody
    hip_yaw_angle = 5
    pb.resetJointState(robot, joint_id["l_hip_aa"], np.radians(hip_yaw_angle),
                       0.)
    pb.resetJointState(robot, joint_id["l_hip_fe"], -np.pi / 4, 0.)
    pb.resetJointState(robot, joint_id["l_knee_fe_jp"], np.pi / 4, 0.)
    pb.resetJointState(robot, joint_id["l_knee_fe_jd"], np.pi / 4, 0.)
    pb.resetJointState(robot, joint_id["l_ankle_fe"], -np.pi / 4, 0.)
    pb.resetJointState(robot, joint_id["l_ankle_ie"],
                       np.radians(-hip_yaw_angle), 0.)

    pb.resetJointState(robot, joint_id["r_hip_aa"], np.radians(-hip_yaw_angle),
                       0.)
    pb.resetJointState(robot, joint_id["r_hip_fe"], -np.pi / 4, 0.)
    pb.resetJointState(robot, joint_id["r_knee_fe_jp"], np.pi / 4, 0.)
    pb.resetJointState(robot, joint_id["r_knee_fe_jd"], np.pi / 4, 0.)
    pb.resetJointState(robot, joint_id["r_ankle_fe"], -np.pi / 4, 0.)
    pb.resetJointState(robot, joint_id["r_ankle_ie"],
                       np.radians(hip_yaw_angle), 0.)


## Run Simulation
if __name__ == "__main__":

    pb.connect(pb.GUI)
    pb.resetDebugVisualizerCamera(cameraDistance=1.5,
                                  cameraYaw=120,
                                  cameraPitch=-30,
                                  cameraTargetPosition=[1, 0.5, 1.5])
    # sim physics setting
    pb.setPhysicsEngineParameter(fixedTimeStep=SimConfig.CONTROLLER_DT,
                                 numSubSteps=SimConfig.N_SUBSTEP)
    pb.setGravity(0, 0, -9.81)

    ## robot spawn & initial kinematics and dynamics setting
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
    fixed_draco = pb.loadURDF(cwd + "/robot_model/draco3_old/draco3_old.urdf",
                              SimConfig.INITIAL_BASE_JOINT_POS,
                              SimConfig.INITIAL_BASE_JOINT_QUAT,
                              useFixedBase=True)

    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        fixed_draco, SimConfig.INITIAL_BASE_JOINT_POS,
        SimConfig.INITIAL_BASE_JOINT_QUAT, SimConfig.PRINT_ROBOT_INFO)

    #robot initial config setting
    set_initial_config(fixed_draco, joint_id)

    #robot joint and link dynamics setting
    pybullet_util.set_joint_friction(fixed_draco, joint_id, 0)
    pybullet_util.set_link_damping(fixed_draco, link_id, 0., 0.)

    #multi-body dynamics library
    robot_sys = PinocchioRobotSystem(
        cwd + "/robot_model/draco3_old/draco3_old.urdf",
        cwd + "/robot_model/draco3_old", True)

    ## rolling contact joint constraint
    c = pb.createConstraint(fixed_draco,
                            link_id['l_knee_fe_lp'],
                            fixed_draco,
                            link_id['l_knee_fe_ld'],
                            jointType=pb.JOINT_GEAR,
                            jointAxis=[0, 1, 0],
                            parentFramePosition=[0, 0, 0],
                            childFramePosition=[0, 0, 0])
    pb.changeConstraint(c, gearRatio=-1, maxForce=500, erp=2)

    c = pb.createConstraint(fixed_draco,
                            link_id['r_knee_fe_lp'],
                            fixed_draco,
                            link_id['r_knee_fe_ld'],
                            jointType=pb.JOINT_GEAR,
                            jointAxis=[0, 1, 0],
                            parentFramePosition=[0, 0, 0],
                            childFramePosition=[0, 0, 0])
    pb.changeConstraint(c, gearRatio=-1, maxForce=500, erp=2)

    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0

    ##get_sensor_data
    pybullet_nominal_sensor_data_dict = pybullet_util.get_sensor_data(
        fixed_draco, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)
    nominal_lf_iso = pybullet_util.get_link_iso(fixed_draco,
                                                link_id['l_foot_contact'])
    nominal_rf_iso = pybullet_util.get_link_iso(fixed_draco,
                                                link_id['r_foot_contact'])
    base_com_pos = np.copy(pybullet_nominal_sensor_data_dict['base_com_pos'])
    base_com_quat = np.copy(pybullet_nominal_sensor_data_dict['base_com_quat'])
    base_joint_pos = pybullet_nominal_sensor_data_dict['base_joint_pos']
    base_joint_quat = pybullet_nominal_sensor_data_dict['base_joint_quat']
    joint_pos = copy.deepcopy(pybullet_nominal_sensor_data_dict['joint_pos'])

    while (True):

        # Get Keyboard Event
        keys = pb.getKeyboardEvents()

        if pybullet_util.is_key_triggered(keys, '8'):
            print("-" * 80)
            print("Pressed 8: ")
            x0 = SX.sym('x0')
            x1 = SX.sym('x1')
            x2 = SX.sym('x2')
            basis_function = x0 * x1 + x2**2
            print(basis_function)

        elif pybullet_util.is_key_triggered(keys, '5'):
            print("-" * 80)
            print("Pressed 5: ")

        elif pybullet_util.is_key_triggered(keys, '1'):
            pass

        # Disable step simulation
        # pb.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1
