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
from itertools import combinations_with_replacement


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


def generate_multivariate_monomials(n_vars, max_degree):
    """
    Generate a CasADi function for multivariate monomial functions up to a given max_degree,
    including all degrees less than the maximum and all combinations of variables.

    Parameters:
    - n_vars: Number of variables in the input vector.
    - max_degree: Maximum degree of the monomials.

    Returns:
    - A CasADi function that computes the monomials for a given input vector.
    - A CasADi function that computes Jacobian for a given input vector. (Jacobian for monomial basis function)
    - The number of monomial basis function
    """
    q = MX.sym('q', n_vars)  # Define the input vector symbolically

    # Function to compute the product of variables for a given combination
    def monomial_combination(comb):
        monomial = 1
        for idx in comb:
            monomial *= q[idx]
        return monomial

    # Generate all combinations of variables and degrees
    monomials = []
    for d in range(1, max_degree + 1):
        for comb in combinations_with_replacement(range(n_vars), d):
            monomials.append(monomial_combination(comb))

    # Create a CasADi function for the monomials
    monomial_func = Function('monomial_func', [q], [vertcat(*monomials)])
    Jacobian_func = Function('Jacobian_func', [q],
                             [jacobian(vertcat(*monomials), q)])

    return monomial_func, Jacobian_func, len(monomials), vertcat(*monomials), q


def sample_random_joint_config(robot_system):
    """
    Sample random joint config using robot model, satisfying joint position limits

    Parameters:
    - robot_system: Robot-specific Rigid body dynamics library instance

    Returns:
    - A CasADi function that computes the monomials for a given input vector.
    """
    joint_pos_limit = robot_system.get_joint_pos_limit
    joint_pos_ll = joint_pos_limit[:, 0]
    joint_pos_ul = joint_pos_limit[:, 1]
    size = joint_pos_ll.shape[0]

    output = np.random.uniform(low=joint_pos_ll, high=joint_pos_ul, size=size)

    #consider rolling contact joint constraints
    robot_joint_id = robot_system.get_joint_id

    l_knee_fe_jp_idx = robot_joint_id["l_knee_fe_jp"]
    l_knee_fe_jd_idx = robot_joint_id["l_knee_fe_jd"]
    r_knee_fe_jp_idx = robot_joint_id["r_knee_fe_jp"]
    r_knee_fe_jd_idx = robot_joint_id["r_knee_fe_jd"]

    output[l_knee_fe_jp_idx] = output[l_knee_fe_jd_idx]
    output[r_knee_fe_jp_idx] = output[r_knee_fe_jd_idx]

    return output


def mirror_sampled_joint(sampled_joint_pos):
    """
    Mirror sampled joint position about sagittal plane

    Parameters:
    - sampled_joint_pos (np.array)

    Returns:
    - mirrored joint pos (np.array).
    """
    mirrored_joint_pos = np.zeros(len(sampled_joint_pos))

    mirrored_joint_pos[:13] = sampled_joint_pos[14:]
    mirrored_joint_pos[13] = sampled_joint_pos[13]
    mirrored_joint_pos[14:] = sampled_joint_pos[:13]

    return np.copy(mirrored_joint_pos)


def quat_misc_mat(quat):
    """
    Parameters
    ----------
    quat (np.array): scalar last quaternion

    Returns
    -------
    ret (np.array): \mathbb{R}^{4\times3} transformation matrix

    """
    quat_xyz = quat[0:3]
    quat_w = quat[3]
    ret = np.zeros((4, 3))
    ret[0, :] = -1 / quat_w * quat_xyz if np.abs(quat_w) > 1e-8 else np.zeros(
        3)
    ret[1:, :] = np.eye(3)
    return np.copy(ret)


## Run Simulation
if __name__ == "__main__":

    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)  #disable camera debugger

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
                              useFixedBase=False)

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
        cwd + "/robot_model/draco3_old", True, False)

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
    nominal_base_com_pos = np.copy(
        pybullet_nominal_sensor_data_dict['base_com_pos'])
    nominal_base_com_quat = np.copy(
        pybullet_nominal_sensor_data_dict['base_com_quat'])
    nominal_base_joint_pos = pybullet_nominal_sensor_data_dict[
        'base_joint_pos']
    nominal_base_joint_quat = pybullet_nominal_sensor_data_dict[
        'base_joint_quat']
    nominal_joint_pos = copy.deepcopy(
        pybullet_nominal_sensor_data_dict['joint_pos'])
    nominal_joint_vel = copy.deepcopy(
        pybullet_nominal_sensor_data_dict['joint_vel'])

    while (True):

        # Get Keyboard Event
        keys = pb.getKeyboardEvents()

        if pybullet_util.is_key_triggered(keys, '1'):
            print("-" * 80)
            print(
                "Pressed 1: test multivariate monomials and its Jacobian function"
            )
            # Example usage
            n_vars = 3  # Number of variables, e.g., x, y, z
            max_degree = 3  # Maximum degree of the monomials

            # Generate the monomial function
            monomial_func, Jacobian_func, size, monomial_basis_symbolic, q = generate_multivariate_monomials(
                n_vars, max_degree)

            # Print out the monomial function vec & its Jacobian
            print("monomial function: ", monomial_func)
            print("Jacobian_func: ", Jacobian_func)
            print("the number of monomial basis function: ", size)
            print("monomial basis symbolic function: ",
                  monomial_basis_symbolic)
            print("symbolic basis: ", q)

            # Example input vector
            x_input = DM([1, 2, 3])

            # Evaluate the monomial function
            monomials_output = monomial_func(x_input)
            print("monomials_output: ", monomials_output)
            print("monomials_output in numpy: ", np.array(monomials_output))
            Jacobian_output = Jacobian_func(x_input)
            print("Jacobian_output: ", Jacobian_output)
            print("Jacobian_output in numpy: ", np.array(Jacobian_output))

        elif pybullet_util.is_key_triggered(keys, '5'):
            print("-" * 80)
            print("Pressed 5: WBO approximation")

            n_vars = 27  # number of draco joint including passive joints
            max_degree = 2  # Maximum degree of the monomials basis function

            # Generate the monomial function and its Jacobian
            monomial_func, jacobian_func, num_basis_func, monomial_basis_symbolic, q = generate_multivariate_monomials(
                n_vars, max_degree)

            # Total number of random joint configs
            N = 2000  # number of random joint config

            # Optimize preparation for WBO Quaternion approximation
            convergence_threshold = 0.01
            cost_val = 1
            num_elements = 3 * num_basis_func  # Total number of elements in the decision variable matrix

            # Define the decision variable as a vector
            theta_vec = MX.sym('theta_vec', num_elements)

            # Reshape the decision variable vector into a matrix for ease of use
            theta_mat = reshape(theta_vec, 3, num_basis_func)

            # Quaternion XYZ approximate with basis functions
            quat_xyz = theta_mat @ monomial_basis_symbolic
            quat_xyz_func = Function('quat_xyz_func', [theta_vec, q],
                                     [quat_xyz])
            theta = np.zeros(
                3 * num_basis_func
            )  #initialize theta decision variables to zero numpy array

            # collect random joint config
            sampled_joint_config_list = []
            for i in range(N):
                joint_pos = sample_random_joint_config(
                    robot_sys) if i % 2 == 0 else mirror_sampled_joint(
                        joint_pos)
                sampled_joint_config_list.append(joint_pos)

                # Set sampled joint config in Pybullet
                joint_pos_dict = robot_sys.create_joint_pos_dict(joint_pos)
                pybullet_util.set_config(fixed_draco, joint_id, link_id,
                                         nominal_base_com_pos,
                                         nominal_base_com_quat, joint_pos_dict)

                # sanity check
                # sensor_data = pybullet_util.get_sensor_data(
                # fixed_draco, joint_id, link_id,
                # pos_basejoint_to_basecom, rot_basejoint_to_basecom)

            # optimization
            objective = 0  # Start with an zero objective function
            # while (cost_val > convergence_threshold):
            total_iter = 3
            for j in range(total_iter):
                for joint_pos in sampled_joint_config_list:
                    # Evaluate WBO Quaternion for the sampled joint config
                    quat_xyz_val = quat_xyz_func(theta,
                                                 joint_pos).full().flatten()
                    w = np.sqrt(max(0, 1 - quat_xyz_val.T.dot(quat_xyz_val)))
                    quat = np.array(
                        [quat_xyz_val[0], quat_xyz_val[1], quat_xyz_val[2], w])
                    normalized_quat = quat / np.linalg.norm(quat)

                    # Calculate T_Q
                    R_Q = util.quat_to_rot(normalized_quat)
                    E_Q = util.quat_rate_to_ang_vel(normalized_quat)
                    K_Q = quat_misc_mat(normalized_quat)
                    T_Q = R_Q @ E_Q @ K_Q

                    # Calculate Jac_lambda
                    J_lambda = jacobian_func(joint_pos).full()

                    # Calculate A matrix
                    joint_pos_dict = robot_sys.create_joint_pos_dict(joint_pos)
                    robot_sys.update_system(
                        nominal_base_com_pos, nominal_base_com_quat,
                        np.zeros(3), np.zeros(3),
                        nominal_base_joint_pos, nominal_base_joint_quat,
                        np.zeros(3), np.zeros(3), joint_pos_dict,
                        nominal_joint_vel, True)

                    Ag = robot_sys.get_Ag
                    A = Ag[:3, :]

                    # Calculate optimization objective function
                    # objective += norm_fro(A - T_Q @ theta_mat @ J_lambda)**2

                    # [alternative] Calculate optimization objective function(Vectorized objective function)
                    objective += norm_fro(
                        DM(reshape(A, -1, 1)) -
                        DM(np.kron(J_lambda.T, T_Q)) @ theta_vec)**2

                #optimization problem setup
                # ipopt
                nlp = {'x': theta_vec, 'f': 1 / N * objective}
                solver = nlpsol('solver', 'ipopt', nlp)
                print(solver)
                result = solver()

                # qpoases
                # qp = {'x': theta_vec, 'f': 1 / N * objective}
                # solver = qpsol('solver', 'qpoases', qp)
                # print(solver)
                # result = solver()

                x_opt = result['x']
                cost_opt = result['f']
                print("=================================================")
                print('x_opt: ', x_opt)
                print('cost_opt: ', cost_opt)
                print("=================================================")

                # update solution
                theta = x_opt

                # update cost
                cost_val = cost_opt

            # C code generation with the optimized theta
            __import__('ipdb').set_trace()
            b_code_gen = True
            if b_code_gen:
                # Define casadi function
                theta[
                    theta <
                    1e-8] = 0  # discard the coefficients that are less than 1e-8
                Q_xyz = reshape(theta, 3,
                                num_basis_func) @ monomial_basis_symbolic
                Q_xyz_func = Function('Q_xyz_func', [q], [Q_xyz])
                Q_xyz_jac_func = Q_xyz_func.jacobian()
                print("=================================================")
                print("C code generation!")
                print(Q_xyz_func)
                print(Q_xyz_jac_func)

                # Code generator
                code_gen = CodeGenerator('draco_wbo_task_helper.c',
                                         dict(with_header=True))
                code_gen.add(Q_xyz_func)
                code_gen.add(Q_xyz_jac_func)
                code_gen.generate()
                print("C code generation done!")
                print("=================================================")

        elif pybullet_util.is_key_triggered(keys, '8'):
            print("-" * 80)
            print("Pressed 8: casadi optimization problem test")

        # Disable step simulation
        # pb.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1
