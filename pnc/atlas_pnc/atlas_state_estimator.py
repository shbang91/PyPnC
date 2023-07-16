import numpy as np
import copy

from config.atlas_config import PnCConfig
from util import util
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider

from pnc.data_saver import DataSaver


class AtlasStateEstimator(object):

    def __init__(self, robot):
        super(AtlasStateEstimator, self).__init__()
        self._robot = robot
        self._sp = AtlasStateProvider(self._robot)
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

        ## save data
        if PnCConfig.SAVE_DATA and (self._sp.count % PnCConfig.SAVE_FREQ == 0):
            self._data_saver.add('joint_pos_act',
                                 copy.deepcopy(sensor_data['joint_pos']))
            self._data_saver.add('joint_vel_act',
                                 copy.deepcopy(sensor_data['joint_vel']))
            self._data_saver.add('lf_contact',
                                 (self._sp.curr_time, self._sp.b_lf_contact))
            self._data_saver.add('rf_contact',
                                 (self._sp.curr_time, self._sp.b_rf_contact))

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
