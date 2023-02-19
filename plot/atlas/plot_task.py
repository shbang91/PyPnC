import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import pickle

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from plot.helper import plot_task, plot_weights, plot_rf_z_max, plot_rf, plot_vector_traj, plot_joints

tasks = [
    'com_pos', 'com_vel', 'pelvis_com_quat', 'pelvis_com_ang_vel',
    'selected_joint_pos', 'selected_joint_vel', 'l_sole_pos', 'l_sole_vel',
    'l_sole_quat', 'l_sole_ang_vel', 'r_sole_pos', 'r_sole_vel', 'r_sole_quat',
    'r_sole_ang_vel'
]

weights = [
    'w_com', 'w_pelvis_com_ori', 'w_selected_joint', 'w_l_sole',
    'w_l_sole_ori', 'w_r_sole', 'w_r_sole_ori'
]

rf_z = ['rf_z_max_r_sole', 'rf_z_max_l_sole']

lfoot_label = [
    'l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky',
    'l_leg_akx'
]

rfoot_label = [
    'r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky',
    'r_leg_akx'
]

time = []

phase = []

rf_cmd = []

pelvis_com_pos = []
pelvis_com_vel = []

des, act = dict(), dict()
for topic in tasks:
    des[topic] = []
    act[topic] = []
w = dict()
for topic in weights:
    w[topic] = []
rf_z_max = dict()
for topic in rf_z:
    rf_z_max[topic] = []

joint_pos_cmd = []
joint_vel_cmd = []
joint_trq_cmd = []

joint_pos_act = []
joint_vel_act = []

joint_label = []

## for task jacobian matrix saving
com_jac = []
pelvis_ori_jac = []
upper_body_jac = []
rfoot_pos_jac = []
lfoot_pos_jac = []
rfoot_ori_jac = []
lfoot_ori_jac = []
rhand_pos_jac = []
lhand_pos_jac = []

## for foot contact info
lf_contact = []
rf_contact = []

with open('data/pnc.pkl', 'rb') as file:
    iter = 0
    while True:
        try:
            d = pickle.load(file)
            if iter == 0:
                joint_label = list(d['joint_pos_des'].keys())
            time.append(d['time'])
            phase.append(d['phase'])
            for topic in tasks:
                des[topic].append(d[topic + '_des'])
                act[topic].append(d[topic])
            for topic in weights:
                w[topic].append(d[topic])
            for topic in rf_z:
                rf_z_max[topic].append(d[topic])
            rf_cmd.append(d['rf_cmd'])
            pelvis_com_pos.append(d['pelvis_com_pos'])
            pelvis_com_vel.append(d['pelvis_com_vel'])
            joint_pos_act.append(list(d['joint_pos_act'].values()))
            joint_vel_act.append(list(d['joint_vel_act'].values()))
            joint_pos_cmd.append(list(d['joint_pos_des'].values()))
            joint_vel_cmd.append(list(d['joint_vel_des'].values()))
            joint_trq_cmd.append(list(d['joint_trq_des'].values()))
            com_jac.append(d['com'])
            pelvis_ori_jac.append(d['pelvis_ori'])
            upper_body_jac.append(d['upper_body'])
            rfoot_pos_jac.append(d['rfoot_pos'])
            lfoot_pos_jac.append(d['lfoot_pos'])
            rfoot_ori_jac.append(d['rfoot_ori'])
            lfoot_ori_jac.append(d['lfoot_ori'])
            rhand_pos_jac.append(d['rhand_pos'])
            lhand_pos_jac.append(d['lhand_pos'])

            lf_contact.append(d['lf_contact'])
            rf_contact.append(d['rf_contact'])

            iter += 1
        except EOFError:
            break

for k, v in des.items():
    des[k] = np.stack(v, axis=0)
for k, v in act.items():
    act[k] = np.stack(v, axis=0)
rf_cmd = np.stack(rf_cmd, axis=0)
pelvis_com_pos = np.stack(pelvis_com_pos, axis=0)
pelvis_com_vel = np.stack(pelvis_com_vel, axis=0)
phase = np.stack(phase, axis=0)

joint_pos_act = np.stack(joint_pos_act, axis=0)
joint_vel_act = np.stack(joint_vel_act, axis=0)
joint_pos_cmd = np.stack(joint_pos_cmd, axis=0)
joint_vel_cmd = np.stack(joint_vel_cmd, axis=0)
joint_trq_cmd = np.stack(joint_trq_cmd, axis=0)

com_jac = np.stack(com_jac, axis=0)
pelvis_ori_jac = np.stack(pelvis_ori_jac, axis=0)
upper_body_jac = np.stack(upper_body_jac, axis=0)
rfoot_pos_jac = np.stack(rfoot_pos_jac, axis=0)
lfoot_pos_jac = np.stack(lfoot_pos_jac, axis=0)
rfoot_ori_jac = np.stack(rfoot_ori_jac, axis=0)
lfoot_ori_jac = np.stack(lfoot_ori_jac, axis=0)
rhand_pos_jac = np.stack(rhand_pos_jac, axis=0)
lhand_pos_jac = np.stack(lhand_pos_jac, axis=0)

lf_contact = np.stack(lf_contact, axis=0)
rf_contact = np.stack(rf_contact, axis=0)

## =============================================================================
## Plot Task
## =============================================================================

plot_task(time, des['com_pos'], act['com_pos'], des['com_vel'], act['com_vel'],
          phase, 'com lin')

plot_task(time, des['pelvis_com_quat'], act['pelvis_com_quat'],
          des['pelvis_com_ang_vel'], act['pelvis_com_ang_vel'], phase,
          'pelvis ori')

plot_task(time, des['selected_joint_pos'], act['selected_joint_pos'],
          des['selected_joint_vel'], act['selected_joint_vel'], phase,
          'upperbody joint')

plot_task(time, des['l_sole_pos'], act['l_sole_pos'], des['l_sole_vel'],
          act['l_sole_vel'], phase, 'left foot lin')

plot_task(time, des['l_sole_quat'], act['l_sole_quat'], des['l_sole_ang_vel'],
          act['l_sole_ang_vel'], phase, 'left foot ori')

plot_task(time, des['r_sole_pos'], act['r_sole_pos'], des['r_sole_vel'],
          act['r_sole_vel'], phase, 'right foot lin')

plot_task(time, des['r_sole_quat'], act['r_sole_quat'], des['r_sole_ang_vel'],
          act['r_sole_ang_vel'], phase, 'right foot ori')

## =============================================================================
## Plot WBC Solutions
## =============================================================================
plot_rf(time, rf_cmd, phase)

plot_joints(joint_label, rfoot_label, time, joint_pos_cmd, joint_pos_act,
            joint_vel_cmd, joint_vel_act, joint_trq_cmd, phase, "rfoot")

plot_joints(joint_label, lfoot_label, time, joint_pos_cmd, joint_pos_act,
            joint_vel_cmd, joint_vel_act, joint_trq_cmd, phase, "lfoot")

## =============================================================================
## Plot Weights and Max Reaction Force Z
## =============================================================================
plot_weights(time, w, phase)

plot_rf_z_max(time, rf_z_max, phase)

plt.show()
