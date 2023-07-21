import numpy as np
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)


class SimConfig(object):
    CONTROLLER_DT = 0.01
    N_SUBSTEP = 10
    CAMERA_DT = 0.05
    KP = 0.
    KD = 0.

    INITIAL_POS_WORLD_TO_BASEJOINT = [0, 0, 1.5 - 0.757]
    INITIAL_QUAT_WORLD_TO_BASEJOINT = [0., 0., 0., 1.]

    PRINT_TIME = False
    PRINT_ROBOT_INFO = True
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


class TowrPlusConfig(object):
    TOWR_PLUS = True
    # SOLUTION_YAML = cwd + "/data/draco3_forward_walk.yaml"
    # SOLUTION_YAML = cwd + "/data/draco3_side_walk.yaml"
    # SOLUTION_YAML = cwd + "/data/draco3_turning.yaml"
    # SOLUTION_YAML = cwd + "/data/draco3_steer_walk.yaml"

    ## TODO:not verified yet ##
    SOLUTION_YAML = cwd + "/data/draco3_stair.yaml"
    # SOLUTION_YAML = cwd + "/data/draco3_block.yaml"
    # SOLUTION_YAML = cwd + "/data/draco3_round_walk.yaml"


class WBCConfig(object):
    VERBOSE = True

    # Max normal force per contact
    RF_Z_MAX = 400.0

    # Task Hierarchy Weights
    W_COM = 30.0
    W_TORSO = 30.0
    W_UPPER_BODY = 20.0
    W_CONTACT_FOOT = 60.0
    W_SWING_FOOT = 80.0

    # Task Gains
    KP_COM = np.array([1000., 1000., 1000])
    KD_COM = np.array([100., 100., 100.])

    KP_TORSO = np.array([1000., 1000., 1000])
    KD_TORSO = np.array([100., 100., 100.])

    # ['neck_pitch', 'l_shoulder_fe', 'l_shoulder_aa', 'l_shoulder_ie',
    # 'l_elbow_fe', 'l_wrist_ps', 'l_wrist_pitch', 'r_shoulder_fe',
    # 'r_shoulder_aa', 'r_shoulder_ie', 'r_elbow_fe', 'r_wrist_ps',
    # 'r_wrist_pitch'
    # ]
    KP_UPPER_BODY = np.array([
        100., 100., 100., 100., 50., 40., 50., 100., 100., 100., 50., 40., 50.
    ])
    KD_UPPER_BODY = np.array(
        [20., 8., 8., 8., 3., 2., 3., 8., 8., 8., 3., 2., 3.])

    KP_FOOT_POS = np.array([500., 500., 500.])
    KD_FOOT_POS = np.array([50., 50., 50.])
    KP_FOOT_ORI = np.array([500., 500., 500.])
    KD_FOOT_ORI = np.array([50., 50., 50.])

    # Regularization terms
    LAMBDA_Q_DDOT = 1e-8
    LAMBDA_RF = 1e-7

    # B_TRQ_LIMIT = True
    B_TRQ_LIMIT = False

    # Integration Parameters
    VEL_CUTOFF_FREQ = 2.0  #Hz
    POS_CUTOFF_FREQ = 1.0  #Hz
    MAX_POS_ERR = 0.2  #Radians


class WalkingConfig(object):
    # STAND
    # INIT_STAND_DUR = 1.0
    INIT_STAND_DUR = 0.5
    RF_Z_MAX_TIME = 0.1

    COM_HEIGHT = 0.67  # m
    # SWING_HEIGHT = 0.03  # m
    SWING_HEIGHT = 0.20  # m

    SWAYING_AMP = np.array([0., 0.08, 0.])
    SWAYING_FREQ = np.array([0., 0.3, 0.])
    # SWAYING_AMP = np.array([0., 0., 0.05])
    # SWAYING_FREQ = np.array([0., 0., 0.3])

    # T_ADDITIONAL_INI_TRANS = 0.  # sec
    # T_CONTACT_TRANS = 1.0
    # T_SWING = 1.0
    # PERCENTAGE_SETTLE = 0.9
    # ALPHA_DS = 0.5

    T_ADDITIONAL_INI_TRANS = 0.  # sec
    T_CONTACT_TRANS = 0.35
    T_SWING = 0.6
    PERCENTAGE_SETTLE = 0.9
    ALPHA_DS = 0.5

    # NOMINAL_FOOTWIDTH = 0.25
    NOMINAL_FOOTWIDTH = 0.202
    NOMINAL_FORWARD_STEP = 0.40
    NOMINAL_BACKWARD_STEP = -0.40
    NOMINAL_TURN_RADIANS = np.pi / 10
    NOMINAL_STRAFE_DISTANCE = 0.4


class WalkingState(object):
    STAND = 0
    BALANCE = 1
    RF_CONTACT_TRANS_START = 2
    RF_CONTACT_TRANS_END = 3
    RF_SWING = 4
    LF_CONTACT_TRANS_START = 5
    LF_CONTACT_TRANS_END = 6
    LF_SWING = 7
    SWAYING = 10
