import numpy as np
from util import util
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

    def __init__(self, robot, tci_container, fm, hm, file):
        self._robot = robot
        self._tci_container = tci_container
        self._fm = fm
        self._hm = hm

        with open(file, 'r') as stream:
            try:
                #Read TOWR Plus solution Trajectory
                data = YAML().load(stream)
                self._time = data["trajectory"]["time"]
                self._base_lin = np.array(data["trajectory"]["base_lin"])
                self._base_ang = np.array(data["trajectory"]["base_ang"])
                self._ee_motion_lin, self._ee_motion_ang, self._ee_wrench_lin, self._ee_wrench_ang, self._contact_schedule = dict(
                ), dict(), dict(), dict(), dict()
                for foot_side in [FootSide.LEFT, FootSide.RIGHT]:
                    self._ee_motion_lin[foot_side] = np.array(
                        data["trajectory"]["ee_motion_lin"][foot_side])
                    self._ee_motion_ang[foot_side] = np.array(
                        data["trajectory"]["ee_motion_ang"][foot_side])
                    self._ee_wrench_lin[foot_side] = np.array(
                        data["trajectory"]["ee_wrench_lin"][foot_side])
                    self._ee_wrench_ang[foot_side] = np.array(
                        data["trajectory"]["ee_wrench_ang"][foot_side])
                    self._contact_schedule[foot_side] = np.array(
                        data["contact_schedule"][foot_side])

                ##print out contact schedule
                print(
                    "=============================================================================================================="
                )
                print(
                    "contact duration for each foot --> 0: Left Foot, 1: Right Foot"
                )
                print(self._contact_schedule)
                print(
                    "=============================================================================================================="
                )

                self._contact_trans_idx = dict()
                for foot_side in [FootSide.LEFT, FootSide.RIGHT]:
                    self._contact_trans_idx[foot_side] = []
                    total_time = 0.
                    for cs in self._contact_schedule[foot_side][:-1]:
                        total_time += cs
                        idx = int(total_time / self._time[1])
                        self._contact_trans_idx[foot_side].append(idx)

            except yaml.YAMLError as exc:
                print(exc)

        self._num_traj_node = len(self._time)
        self._iter = 0
        self._trans_dur = 10

        self._trans_start_iter = 0
        self._b_lfoot_trans_sw = False
        self._b_lfoot_trans_ld = False
        self._b_rfoot_trans_sw = False
        self._b_rfoot_trans_ld = False

    def update_desired(self):
        ## update desired task traj
        if self._iter < self._num_traj_node:
            ## CoM
            com_lin_pos = self._base_lin[self._iter, 0:3]
            com_lin_vel = self._base_lin[self._iter, 3:6]
            self._tci_container.com_task.update_desired(
                com_lin_pos, com_lin_vel, np.zeros(3))

            ## pelvis ori
            base_ang_quat = util.rot_to_quat(
                util.euler_to_rot(self._base_ang[self._iter, 0:3]))
            base_ang_vel = util.euler_rates_to_ang_vel(
                self._base_ang[self._iter, 0:3], self._base_ang[self._iter,
                                                                3:6])
            self._tci_container.torso_ori_task.update_desired(
                base_ang_quat, base_ang_vel, np.zeros(3))

            ## Right foot
            rfoot_lin_pos = self._ee_motion_lin[FootSide.RIGHT][self._iter,
                                                                0:3]
            rfoot_lin_vel = self._ee_motion_lin[FootSide.RIGHT][self._iter,
                                                                3:6]
            self._tci_container.rfoot_pos_task.update_desired(
                rfoot_lin_pos, rfoot_lin_vel, np.zeros(3))

            rfoot_ang_quat = util.rot_to_quat(
                util.euler_to_rot(
                    self._ee_motion_ang[FootSide.RIGHT][self._iter, 0:3]))
            rfoot_ang_vel = util.euler_rates_to_ang_vel(
                self._ee_motion_ang[FootSide.RIGHT][self._iter, 0:3],
                self._ee_motion_ang[FootSide.RIGHT][self._iter, 3:6])
            self._tci_container.rfoot_ori_task.update_desired(
                rfoot_ang_quat, rfoot_ang_vel, np.zeros(3))

            ## Left foot
            lfoot_lin_pos = self._ee_motion_lin[FootSide.LEFT][self._iter, 0:3]
            lfoot_lin_vel = self._ee_motion_lin[FootSide.LEFT][self._iter, 3:6]
            self._tci_container.lfoot_pos_task.update_desired(
                lfoot_lin_pos, lfoot_lin_vel, np.zeros(3))

            lfoot_ang_quat = util.rot_to_quat(
                util.euler_to_rot(
                    self._ee_motion_ang[FootSide.LEFT][self._iter, 0:3]))
            lfoot_ang_vel = util.euler_rates_to_ang_vel(
                self._ee_motion_ang[FootSide.LEFT][self._iter, 0:3],
                self._ee_motion_ang[FootSide.LEFT][self._iter, 3:6])
            self._tci_container.lfoot_ori_task.update_desired(
                lfoot_ang_quat, lfoot_ang_vel, np.zeros(3))

            ## TODO:contact wrench task update

            ## initialize reaction force managers & foot task hierarchy managers
            if self._iter in [
                    lf_contact_sw - self._trans_dur for lf_contact_sw in
                    self._contact_trans_idx[FootSide.LEFT][::2]
            ]:
                self._fm["lfoot"].initialize_ramp_to_min(
                    self._iter, self._trans_dur)
                self._hm["lfoot_pos"].initialize_ramp_to_min(
                    self._iter, self._trans_dur)
                self._hm["lfoot_ori"].initialize_ramp_to_min(
                    self._iter, self._trans_dur)
                self._trans_start_iter = self._iter
                self._b_lfoot_trans_sw = True

            if self._iter in [
                    lf_contact_ld for lf_contact_ld in self._contact_trans_idx[
                        FootSide.LEFT][1::2]
            ]:
                self._fm["lfoot"].initialize_ramp_to_max(
                    self._iter, self._trans_dur)
                self._hm["lfoot_pos"].initialize_ramp_to_max(
                    self._iter, self._trans_dur)
                self._hm["lfoot_ori"].initialize_ramp_to_max(
                    self._iter, self._trans_dur)
                self._trans_start_iter = self._iter
                self._b_lfoot_trans_ld = True

            if self._iter in [
                    rf_contact_sw - self._trans_dur for rf_contact_sw in
                    self._contact_trans_idx[FootSide.RIGHT][::2]
            ]:
                self._fm["rfoot"].initialize_ramp_to_min(
                    self._iter, self._trans_dur)
                self._hm["rfoot_pos"].initialize_ramp_to_min(
                    self._iter, self._trans_dur)
                self._hm["rfoot_ori"].initialize_ramp_to_min(
                    self._iter, self._trans_dur)
                self._trans_start_iter = self._iter
                self._b_rfoot_trans_sw = True

            if self._iter in [
                    rf_contact_ld for rf_contact_ld in self._contact_trans_idx[
                        FootSide.RIGHT][1::2]
            ]:
                self._fm["rfoot"].initialize_ramp_to_max(
                    self._iter, self._trans_dur)
                self._hm["rfoot_pos"].initialize_ramp_to_max(
                    self._iter, self._trans_dur)
                self._hm["rfoot_ori"].initialize_ramp_to_max(
                    self._iter, self._trans_dur)
                self._trans_start_iter = self._iter
                self._b_rfoot_trans_ld = True

            ## update reaction force and task hierarchy managers
            if self._iter >= self._trans_start_iter and self._iter <= self._trans_start_iter + self._trans_dur and self._b_lfoot_trans_sw:
                self._fm["lfoot"].update_ramp_to_min(self._iter)
                self._hm["lfoot_pos"].update_ramp_to_min(self._iter)
                self._hm["lfoot_ori"].update_ramp_to_min(self._iter)
                if self._iter == self._trans_start_iter + self._trans_dur:
                    self._b_lfoot_trans_sw = False

            if self._iter >= self._trans_start_iter and self._iter <= self._trans_start_iter + self._trans_dur and self._b_lfoot_trans_ld:
                self._fm["lfoot"].update_ramp_to_max(self._iter)
                self._hm["lfoot_pos"].update_ramp_to_max(self._iter)
                self._hm["lfoot_ori"].update_ramp_to_max(self._iter)
                if self._iter == self._trans_start_iter + self._trans_dur:
                    self._b_lfoot_trans_ld = False

            if self._iter >= self._trans_start_iter and self._iter <= self._trans_start_iter + self._trans_dur and self._b_rfoot_trans_sw:
                self._fm["rfoot"].update_ramp_to_min(self._iter)
                self._hm["rfoot_pos"].update_ramp_to_min(self._iter)
                self._hm["rfoot_ori"].update_ramp_to_min(self._iter)
                if self._iter == self._trans_start_iter + self._trans_dur:
                    self._b_rfoot_trans_sw = False

            if self._iter >= self._trans_start_iter and self._iter <= self._trans_start_iter + self._trans_dur and self._b_rfoot_trans_ld:
                self._fm["rfoot"].update_ramp_to_max(self._iter)
                self._hm["rfoot_pos"].update_ramp_to_max(self._iter)
                self._hm["rfoot_ori"].update_ramp_to_max(self._iter)
                if self._iter == self._trans_start_iter + self._trans_dur:
                    self._b_rfoot_trans_ld = False

            self._iter += 1
