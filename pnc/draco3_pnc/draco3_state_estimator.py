import numpy as np
import copy

from config.draco3_config import PnCConfig
from util import util
from pnc.draco3_pnc.draco3_state_provider import Draco3StateProvider

from pnc.data_saver import DataSaver


class Draco3StateEstimator(object):

    def __init__(self, robot):
        super(Draco3StateEstimator, self).__init__()
        self._robot = robot
        self._sp = Draco3StateProvider(self._robot)
        self._data_saver = DataSaver()

    def initialize(self, sensor_data):
        self._sp.nominal_joint_pos = sensor_data["joint_pos"]

    def update(self, sensor_data):

        # Update Encoders
        self._robot.update_system(
            sensor_data["base_com_pos"], sensor_data["base_com_quat"],
            sensor_data["base_com_lin_vel"], sensor_data["base_com_ang_vel"],
            sensor_data["base_joint_pos"], sensor_data["base_joint_quat"],
            sensor_data["base_joint_lin_vel"],
            sensor_data["base_joint_ang_vel"], sensor_data["joint_pos"],
            sensor_data["joint_vel"])

        # Update Contact Info
        self._sp.b_rf_contact = sensor_data["b_rf_contact"]
        self._sp.b_lf_contact = sensor_data["b_lf_contact"]

        # Update Divergent Component of Motion
        self._update_dcm()

        # change pybullet joint order to dynamics library model order
        sensor_data_pos_vel = self._robot.create_sensor_data_ordered_dict(
            copy.deepcopy(sensor_data['joint_pos']),
            copy.deepcopy(sensor_data['joint_vel']))

        if PnCConfig.SAVE_DATA:
            self._data_saver.add(
                'joint_pos_act',
                copy.deepcopy(sensor_data_pos_vel['joint_pos']))
            self._data_saver.add(
                'joint_vel_act',
                copy.deepcopy(sensor_data_pos_vel['joint_vel']))

    def _update_dcm(self):
        com_pos = self._robot.get_com_pos()
        com_vel = self._robot.get_com_lin_vel()
        dcm_omega = np.sqrt(9.81 / com_pos[2])
        self._sp.prev_dcm = np.copy(self._sp.dcm)
        self._sp.dcm = com_pos + com_vel / dcm_omega
        alpha_dcm_vel = 0.1  # TODO : Get this from Hz
        self._sp.dcm_vel = alpha_dcm_vel * (
            self._sp.dcm - self._sp.prev_dcm) / PnCConfig.CONTROLLER_DT
        +(1.0 - alpha_dcm_vel) * self._sp.dcm_vel
