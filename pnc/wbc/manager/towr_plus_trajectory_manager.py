import numpy as np
from ruamel.yaml import YAML

class FootSide(object):
    LEFT = 0
    RIGHT = 1

class TowrPlusTrajectoryManager(object):
    """
    TOWR Plus Trajectory Manager
    -----------------------------
    Usage:
        -> update_desried()
    """

    def __init__(self, robot, tci_container, file):
        self._robot = robot
        self._tci_container = tci_container

        with open(file, 'r') as stream:
            try:
                #Read TOWR Plus solution Trajectory
                data = YAML().load(stream)
                self._time = data["trajectory"]["time"]
                self._base_lin = np.array(data["trajectory"]["base_lin"])
                self._base_ang = np.array(data["trajectory"]["base_ang"])
                self._ee_motion_lin, self._ee_motion_ang, self._ee_wrench_lin, self._ee_wrench_ang, self._contact_schedule = dict(), dict(), dict(), dict(), dict()
                for foot_side in [FootSide.LEFT, FootSide.RIGHT]:
                    self._ee_motion_lin[foot_side] = np.array(data["trajectory"]["ee_motion_lin"][foot_side])
                    self._ee_motion_ang[foot_side] = np.array(data["trajectory"]["ee_motion_ang"][foot_side])
                    self._ee_wrench_lin[foot_side] = np.array(data["trajectory"]["ee_wrench_lin"][foot_side])
                    self._ee_wrench_ang[foot_side] = np.array(data["trajectory"]["ee_wrench_ang"][foot_side])
                    self._contact_schedule[foot_side] = np.array(data["contact_schedule"][foot_side])
            except yaml.YAMLError as exc:
                print(exc)

    def update_desired(self):
        ## update desired task traj
        pass

