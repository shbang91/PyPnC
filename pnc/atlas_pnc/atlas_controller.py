import numpy as np
import copy

from util import util
from config.atlas_config import PnCConfig, WBCConfig
from pnc.wbc.ihwbc.ihwbc import IHWBC
from pnc.wbc.ihwbc.joint_integrator import JointIntegrator
from pnc.data_saver import DataSaver
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider


class AtlasController(object):
    def __init__(self, tci_container, robot):
        self._tci_container = tci_container
        self._robot = robot

        # Initialize WBC
        act_list = [False] * robot.n_floating + [True] * robot.n_a
        n_q_dot = len(act_list)
        n_active = np.count_nonzero(np.array(act_list))
        n_passive = n_q_dot - n_active - 6

        self._sa = np.zeros((n_active, n_q_dot))
        self._sv = np.zeros((n_passive, n_q_dot))
        j, k = 0, 0
        for i in range(n_q_dot):
            if i >= 6:
                if act_list[i]:
                    self._sa[j, i] = 1.
                    j += 1
                else:
                    self._sv[k, i] = 1.
                    k += 1
        self._sf = np.zeros((6, n_q_dot))
        self._sf[0:6, 0:6] = np.eye(6)

        self._ihwbc = IHWBC(self._sf, self._sa, self._sv, PnCConfig.SAVE_DATA)
        if WBCConfig.B_TRQ_LIMIT:
            self._ihwbc.trq_limit = np.dot(self._sa[:, 6:],
                                           self._robot.joint_trq_limit)
        self._ihwbc.lambda_q_ddot = WBCConfig.LAMBDA_Q_DDOT
        self._ihwbc.lambda_rf = WBCConfig.LAMBDA_RF
        # Initialize Joint Integrator
        self._joint_integrator = JointIntegrator(robot.n_a,
                                                 PnCConfig.CONTROLLER_DT)
        self._joint_integrator.pos_cutoff_freq = WBCConfig.POS_CUTOFF_FREQ
        self._joint_integrator.vel_cutoff_freq = WBCConfig.VEL_CUTOFF_FREQ
        self._joint_integrator.max_pos_err = WBCConfig.MAX_POS_ERR
        self._joint_integrator.joint_pos_limit = self._robot.joint_pos_limit
        self._joint_integrator.joint_vel_limit = self._robot.joint_vel_limit

        self._b_first_visit = True

        self._sp = AtlasStateProvider()

        if PnCConfig.SAVE_DATA:
            self._data_saver = DataSaver()

    def get_command(self):

        if self._b_first_visit:
            self.first_visit()

        # Dynamics properties
        mass_matrix = self._robot.get_mass_matrix()
        mass_matrix_inv = np.linalg.inv(mass_matrix)
        coriolis = self._robot.get_coriolis()
        gravity = self._robot.get_gravity()
        self._ihwbc.update_setting(mass_matrix, mass_matrix_inv, coriolis,
                                   gravity)
        # Task, Contact, and Internal Constraint Setup
        w_hierarchy_list = []
        for [task_str, task] in self._tci_container.task_list.items():
            task.update_jacobian()
            task.update_cmd()
            w_hierarchy_list.append(task.w_hierarchy)

            if PnCConfig.SAVE_DATA and (self._sp.count % PnCConfig.SAVE_FREQ
                                        == 0):
                self._data_saver.add(task_str, task.jacobian)

        self._ihwbc.w_hierarchy = np.array(w_hierarchy_list)
        for contact in self._tci_container.contact_list:
            contact.update_contact()
        for internal_constraint in self._tci_container.internal_constraint_list:
            internal_constraint.update_internal_constraint()
        # WBC commands
        joint_trq_cmd, joint_acc_cmd, rf_cmd = self._ihwbc.solve(
            self._tci_container.task_list, self._tci_container.contact_list,
            self._tci_container.internal_constraint_list)
        # Double integration
        joint_vel_cmd, joint_pos_cmd = self._joint_integrator.integrate(
            joint_acc_cmd, self._robot.joint_velocities,
            self._robot.joint_positions)

        command = self._robot.create_cmd_ordered_dict(joint_pos_cmd,
                                                      joint_vel_cmd,
                                                      joint_trq_cmd)
        ## save wbc data
        if PnCConfig.SAVE_DATA and (self._sp.count % PnCConfig.SAVE_FREQ == 0):
            self._data_saver.add('joint_pos_des',
                                 copy.deepcopy(command['joint_pos']))
            self._data_saver.add('joint_vel_des',
                                 copy.deepcopy(command['joint_vel']))
            self._data_saver.add('joint_trq_des',
                                 copy.deepcopy(command['joint_trq']))
            self._data_saver.add('pelvis_com_pos',
                                 self._robot.get_link_iso('pelvis_com')[:3, 3])
            self._data_saver.add('pelvis_com_vel',
                                 self._robot.get_link_vel('pelvis_com')[3:6])

        return command

    def first_visit(self):
        joint_pos_ini = self._robot.joint_positions
        self._joint_integrator.initialize_states(np.zeros(self._robot.n_a),
                                                 joint_pos_ini)

        self._b_first_visit = False
