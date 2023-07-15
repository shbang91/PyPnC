import numpy as np

from config.atlas_config import WalkingConfig, WBCConfig, WalkingState
from pnc.control_architecture import ControlArchitecture
from pnc.planner.locomotion.dcm_planner.dcm_planner import DCMPlanner
from pnc.planner.locomotion.dcm_planner.footstep import Footstep
from pnc.wbc.manager.dcm_trajectory_manager import DCMTrajectoryManager
from pnc.wbc.manager.task_hierarchy_manager import TaskHierarchyManager
from pnc.wbc.manager.hand_trajectory_manager import HandTrajectoryManager
from pnc.wbc.manager.floating_base_pelvis_trajectory_manager import FloatingBasePelvisTrajectoryManager
from pnc.wbc.manager.foot_trajectory_manager import FootTrajectoryManager
from pnc.wbc.manager.reaction_force_manager import ReactionForceManager
from pnc.wbc.manager.upper_body_trajectory_manager import UpperBodyTrajectoryManager
from pnc.atlas_pnc.atlas_tci_container import AtlasTCIContainer
from pnc.atlas_pnc.atlas_controller import AtlasController
from pnc.atlas_pnc.atlas_state_machine.double_support_stand import DoubleSupportStand
from pnc.atlas_pnc.atlas_state_machine.double_support_balance import DoubleSupportBalance
from pnc.atlas_pnc.atlas_state_machine.contact_transition_start import ContactTransitionStart
from pnc.atlas_pnc.atlas_state_machine.contact_transition_end import ContactTransitionEnd
from pnc.atlas_pnc.atlas_state_machine.single_support_swing import SingleSupportSwing
from pnc.atlas_pnc.atlas_state_machine.double_support_hand_reaching import DoubleSupportHandReach
from pnc.atlas_pnc.atlas_state_provider import AtlasStateProvider


class AtlasControlArchitecture(ControlArchitecture):

    def __init__(self, robot):
        super(AtlasControlArchitecture, self).__init__(robot)

        # Initialize Task Force Container
        self._tci_container = AtlasTCIContainer(robot)

        # Initialize Controller
        self._atlas_controller = AtlasController(self._tci_container, robot)

        # Initialize Planner
        self._dcm_planner = DCMPlanner()

        # Initialize Task Manager
        self._rfoot_tm = FootTrajectoryManager(
            self._tci_container.rfoot_pos_task,
            self._tci_container.rfoot_ori_task, robot)
        self._rfoot_tm.swing_height = WalkingConfig.SWING_HEIGHT
        self._lfoot_tm = FootTrajectoryManager(
            self._tci_container.lfoot_pos_task,
            self._tci_container.lfoot_ori_task, robot)
        self._lfoot_tm.swing_height = WalkingConfig.SWING_HEIGHT
        self._upper_body_tm = UpperBodyTrajectoryManager(
            self._tci_container.upper_body_task, robot)
        self._floating_base_pelvis_tm = FloatingBasePelvisTrajectoryManager(
            self._tci_container.com_task, self._tci_container.torso_ori_task,
            self._tci_container.pelvis_ori_task, robot)

        self._dcm_tm = DCMTrajectoryManager(self._dcm_planner,
                                            self._tci_container.com_task,
                                            self._tci_container.torso_ori_task,
                                            self._robot, "l_sole", "r_sole")
        self._dcm_tm.nominal_com_height = WalkingConfig.COM_HEIGHT
        self._dcm_tm.t_additional_init_transfer = WalkingConfig.T_ADDITIONAL_INI_TRANS
        self._dcm_tm.t_contact_transition = WalkingConfig.T_CONTACT_TRANS
        self._dcm_tm.t_swing = WalkingConfig.T_SWING
        self._dcm_tm.percentage_settle = WalkingConfig.PERCENTAGE_SETTLE
        self._dcm_tm.alpha_ds = WalkingConfig.ALPHA_DS
        self._dcm_tm.nominal_footwidth = WalkingConfig.NOMINAL_FOOTWIDTH
        self._dcm_tm.nominal_forward_step = WalkingConfig.NOMINAL_FORWARD_STEP
        self._dcm_tm.nominal_backward_step = WalkingConfig.NOMINAL_BACKWARD_STEP
        self._dcm_tm.nominal_turn_radians = WalkingConfig.NOMINAL_TURN_RADIANS
        self._dcm_tm.nominal_strafe_distance = WalkingConfig.NOMINAL_STRAFE_DISTANCE

        self._lhand_tm = HandTrajectoryManager(
            self._tci_container.lhand_pos_task, None, robot)

        self._rhand_tm = HandTrajectoryManager(
            self._tci_container.rhand_pos_task, None, robot)

        self._trajectory_managers = {
            "rfoot": self._rfoot_tm,
            "lfoot": self._lfoot_tm,
            "upper_body": self._upper_body_tm,
            "floating_base": self._floating_base_pelvis_tm,
            "dcm": self._dcm_tm,
            "lhand": self._lhand_tm,
            "rhand": self._rhand_tm
        }

        # Initialize Hierarchy Manager
        self._rfoot_pos_hm = TaskHierarchyManager(
            self._tci_container.rfoot_pos_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)
        self._lfoot_pos_hm = TaskHierarchyManager(
            self._tci_container.lfoot_pos_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)
        self._rfoot_ori_hm = TaskHierarchyManager(
            self._tci_container.rfoot_ori_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)
        self._lfoot_ori_hm = TaskHierarchyManager(
            self._tci_container.lfoot_ori_task, WBCConfig.W_CONTACT_FOOT,
            WBCConfig.W_SWING_FOOT)
        self._lhand_pos_hm = TaskHierarchyManager(
            self._tci_container.lhand_pos_task, WBCConfig.W_HAND_POS_MAX,
            WBCConfig.W_HAND_POS_MIN)
        self._rhand_pos_hm = TaskHierarchyManager(
            self._tci_container.rhand_pos_task, WBCConfig.W_HAND_POS_MAX,
            WBCConfig.W_HAND_POS_MIN)
        self._hierarchy_managers = {
            "rfoot_pos": self._rfoot_pos_hm,
            "lfoot_pos": self._lfoot_pos_hm,
            "rfoot_ori": self._rfoot_ori_hm,
            "lfoot_ori": self._lfoot_ori_hm,
            "lhand_pos": self._lhand_pos_hm,
            "rhand_pos": self._rhand_pos_hm,
        }

        # Initialize Reaction Force Manager
        self._rfoot_fm = ReactionForceManager(
            self._tci_container.rfoot_contact, WBCConfig.RF_Z_MAX)
        self._lfoot_fm = ReactionForceManager(
            self._tci_container.lfoot_contact, WBCConfig.RF_Z_MAX)
        self._reaction_force_managers = {
            "rfoot": self._rfoot_fm,
            "lfoot": self._lfoot_fm
        }

        # Initialize State Machines
        self._state_machine[WalkingState.STAND] = DoubleSupportStand(
            WalkingState.STAND, self._trajectory_managers,
            self._hierarchy_managers, self._reaction_force_managers, robot)
        self._state_machine[
            WalkingState.STAND].end_time = WalkingConfig.INIT_STAND_DUR
        self._state_machine[
            WalkingState.STAND].rf_z_max_time = WalkingConfig.RF_Z_MAX_TIME
        self._state_machine[
            WalkingState.STAND].com_height_des = WalkingConfig.COM_HEIGHT

        self._state_machine[WalkingState.BALANCE] = DoubleSupportBalance(
            WalkingState.BALANCE, self._trajectory_managers,
            self._hierarchy_managers, self._reaction_force_managers, robot)

        self._state_machine[
            WalkingState.LF_CONTACT_TRANS_START] = ContactTransitionStart(
                WalkingState.LF_CONTACT_TRANS_START, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                Footstep.LEFT_SIDE, self._robot)
        self._state_machine[
            WalkingState.LF_CONTACT_TRANS_END] = ContactTransitionEnd(
                WalkingState.LF_CONTACT_TRANS_END, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                Footstep.LEFT_SIDE, self._robot)
        self._state_machine[WalkingState.LF_SWING] = SingleSupportSwing(
            WalkingState.LF_SWING, self._trajectory_managers,
            Footstep.LEFT_SIDE, self._robot)

        self._state_machine[
            WalkingState.RF_CONTACT_TRANS_START] = ContactTransitionStart(
                WalkingState.RF_CONTACT_TRANS_START, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                Footstep.RIGHT_SIDE, self._robot)
        self._state_machine[
            WalkingState.RF_CONTACT_TRANS_END] = ContactTransitionEnd(
                WalkingState.RF_CONTACT_TRANS_END, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                Footstep.RIGHT_SIDE, self._robot)
        self._state_machine[WalkingState.RF_SWING] = SingleSupportSwing(
            WalkingState.RF_SWING, self._trajectory_managers,
            Footstep.RIGHT_SIDE, self._robot)

        self._state_machine[
            WalkingState.RH_HANDREACH] = DoubleSupportHandReach(
                WalkingState.RH_HANDREACH, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                self._robot)
        self._state_machine[
            WalkingState.
            RH_HANDREACH].moving_duration = WalkingConfig.T_REACHING_DURATION
        self._state_machine[
            WalkingState.
            RH_HANDREACH].rh_target_pos = WalkingConfig.RH_TARGET_POS
        self._state_machine[
            WalkingState.
            RH_HANDREACH].trans_duration = WalkingConfig.T_TRANS_DURATION

        self._state_machine[
            WalkingState.LH_HANDREACH] = DoubleSupportHandReach(
                WalkingState.LH_HANDREACH, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers,
                self._robot)
        self._state_machine[
            WalkingState.
            LH_HANDREACH].moving_duration = WalkingConfig.T_REACHING_DURATION
        self._state_machine[
            WalkingState.
            LH_HANDREACH].lh_target_pos = WalkingConfig.LH_TARGET_POS
        self._state_machine[
            WalkingState.
            LH_HANDREACH].trans_duration = WalkingConfig.T_TRANS_DURATION

        # Set Starting State
        self._state = WalkingState.STAND
        self._prev_state = WalkingState.STAND
        self._b_state_first_visit = True

        self._sp = AtlasStateProvider()

    def get_command(self):
        if self._b_state_first_visit:
            self._state_machine[self._state].first_visit()
            self._b_state_first_visit = False

        # Update State Machine
        self._state_machine[self._state].one_step()
        # Update State Machine Independent Trajectories
        self._upper_body_tm.use_nominal_upper_body_joint_pos(
            self._sp.nominal_joint_pos)
        # Get Whole Body Control Commands
        command = self._atlas_controller.get_command()

        if self._state_machine[self._state].end_of_state():
            self._state_machine[self._state].last_visit()
            self._prev_state = self._state
            self._state = self._state_machine[self._state].get_next_state()
            self._b_state_first_visit = True

        return command

    @property
    def dcm_tm(self):
        return self._dcm_tm

    @property
    def state_machine(self):
        return self._state_machine
