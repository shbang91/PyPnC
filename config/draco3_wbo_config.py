import numpy as np
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)


class SimConfig(object):
    CONTROLLER_DT = 0.001
    N_SUBSTEP = 1
    KP = 0.
    KD = 0.

    INITIAL_BASE_JOINT_POS = [0, 0, 1]
    INITIAL_BASE_JOINT_QUAT = [0., 0., 0., 1.]

    PRINT_TIME = False
    PRINT_ROBOT_INFO = False
    VIDEO_RECORD = False
    RECORD_FREQ = 5
    SIMULATE_CAMERA = False
    SAVE_CAMERA_DATA = False


class PnCConfig(object):
    DYN_LIB = "pinocchio"  # "dart"
    CONTROLLER_DT = SimConfig.CONTROLLER_DT
    SAVE_DATA = True
    SAVE_FREQ = 1

    PRINT_ROBOT_INFO = SimConfig.PRINT_ROBOT_INFO
