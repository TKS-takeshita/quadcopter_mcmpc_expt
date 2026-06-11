import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection


default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_20260611_160418.csv"

# ===== basic settings =====
dt = 0.02
horizon = 75
prediction_start_time = 0.1
prediction_interval = 1.5

SQUARE_Z = -1.0
SQUARE_WAYPOINT_THRESHOLD = 0.15
SQUARE_WAYPOINT_HOLD_SEC = 1.0
SQUARE_WAYPOINTS = np.array([
    [1.0, 0.0, SQUARE_Z],
    [1.0, 1.0, SQUARE_Z],
    [-1.0, 1.0, SQUARE_Z],
    [-1.0, -1.0, SQUARE_Z],
    [1.0, -1.0, SQUARE_Z],
    [1.0, 0.0, SQUARE_Z],
], dtype=float)
SQUARE_START_TARGET = np.array([0.0, 0.0, -0.8], dtype=float)

SIMULATION = True
# SIMULATION = False

# ===== PX4/MPC parameters =====
if SIMULATION:
    A_OF_GRAVITY = 9.80665
    MAX_THRUST = 38.42

    MPC_XY_P = 0.30
    MPC_Z_P = 1.00

    MPC_XY_VEL_P_ACC = 1.80
    MPC_XY_VEL_I_ACC = 0.40
    MPC_XY_VEL_D_ACC = 0.20

    MPC_Z_VEL_P_ACC = 4.00
    MPC_Z_VEL_I_ACC = 2.00
    MPC_Z_VEL_D_ACC = 0.00

    MPC_XY_VEL_MAX = 12.0
    MPC_Z_VEL_MAX_UP = 3.0
    MPC_Z_VEL_MAX_DOWN = 1.0

    MPC_THR_HOVER = 0.60
    MPC_THR_MIN = 0.1
    MPC_THR_MAX = 0.9
    MPC_THR_XY_MARGIN = 0.3
    MPC_TILT_MAX = np.deg2rad(45.0)

    MC_ROLL_P = 3.300
    MC_PITCH_P = 3.300
    MC_YAW_P = 2.80

    MPC_VELD_LP = 5.0
    ARW_GAIN = 2.0 / MPC_XY_VEL_P_ACC

else:
    A_OF_GRAVITY = 9.80665
    MAX_THRUST = 38.42

    MPC_XY_P = 0.950
    MPC_Z_P = 1.00

    MPC_XY_VEL_P_ACC = 1.80
    MPC_XY_VEL_I_ACC = 0.40
    MPC_XY_VEL_D_ACC = 0.20

    MPC_Z_VEL_P_ACC = 4.00
    MPC_Z_VEL_I_ACC = 2.00
    MPC_Z_VEL_D_ACC = 0.00

    MPC_XY_VEL_MAX = 12.0
    MPC_Z_VEL_MAX_UP = 3.0
    MPC_Z_VEL_MAX_DOWN = 1.0

    MPC_THR_HOVER = 0.65
    MPC_THR_MIN = 0.1
    MPC_THR_MAX = 0.9
    MPC_THR_XY_MARGIN = 0.3
    MPC_TILT_MAX = np.deg2rad(45.0)

    MC_ROLL_P = 4.00
    MC_PITCH_P = 4.00
    MC_YAW_P = 2.80

    MPC_VELD_LP = 5.0
    ARW_GAIN = 2.0 / MPC_XY_VEL_P_ACC
    

# ===== identified internal model in mpc_simulator.cu =====
MODEL_VEL_A = np.array([1.0096476, 1.0116026, 0.99311937])
MODEL_VEL_B = np.array([0.012726781, 0.013022681, 0.00016697574])
MODEL_W_C = np.array([0.81389003, 0.81882324, 0.99194815])
MODEL_W_D = np.array([0.12783148, 0.10071088, -6.9079037e-05])
MODEL_POS_ALPHA = np.array([0.46913642, 0.47037031, 0.48688093])
MODEL_ATT_BETA = 0.41367774


state_names = [
    "e0", "e1", "e2", "e3",
    "wx", "wy", "wz",
    "x", "y", "z",
    "vx", "vy", "vz",
]
state_index = {name: i for i, name in enumerate(state_names)}

rate_names = ["roll_rate", "pitch_rate", "yaw_rate"]
rate_cols = ["px4_sp_roll_rate", "px4_sp_pitch_rate", "px4_sp_yaw_rate"]
acc_names = ["acc_x", "acc_y", "acc_z"]
acc_cols = ["px4_sp_acc_x", "px4_sp_acc_y", "px4_sp_acc_z"]
setpoint_group_names = ["acc_setpoint", "omega_setpoint"]
acc_index = {name: i for i, name in enumerate(acc_names)}
rate_alias = {"wx_sp": "roll_rate", "wy_sp": "pitch_rate", "wz_sp": "yaw_rate"}
rate_index = {name: i for i, name in enumerate(rate_names)}
rate_to_state = {"roll_rate": "wx", "pitch_rate": "wy", "yaw_rate": "wz"}
state_to_rate = {"wx": "roll_rate", "wy": "pitch_rate", "wz": "yaw_rate"}


class RLSModel:
    def __init__(self, theta, lower=None, upper=None, p_scale=1000.0, forgetting=0.995):
        self.theta = np.asarray(theta, dtype=float)
        self.p = np.eye(len(self.theta)) * p_scale
        self.forgetting = forgetting
        self.lower = lower
        self.upper = upper

    def predict(self, phi):
        phi = np.asarray(phi, dtype=float)
        if not np.all(np.isfinite(phi)):
            return np.nan
        return float(phi @ self.theta)

    def update(self, phi, y):
        phi = np.asarray(phi, dtype=float)
        if not np.isfinite(y) or not np.all(np.isfinite(phi)):
            return

        p_phi = self.p @ phi
        denom = self.forgetting + phi @ p_phi
        if abs(denom) < 1e-9:
            return

        gain = p_phi / denom
        err = y - phi @ self.theta
        self.theta = self.theta + gain * err
        if self.lower is not None or self.upper is not None:
            lo = -np.inf if self.lower is None else self.lower
            hi = np.inf if self.upper is None else self.upper
            self.theta = np.clip(self.theta, lo, hi)
        self.p = (self.p - np.outer(gain, phi) @ self.p) / self.forgetting


def apply_param_profile(profile):
    global MPC_XY_P, MPC_Z_P
    global MPC_XY_VEL_P_ACC, MPC_XY_VEL_I_ACC, MPC_XY_VEL_D_ACC
    global MPC_Z_VEL_P_ACC, MPC_Z_VEL_I_ACC, MPC_Z_VEL_D_ACC
    global MC_ROLL_P, MC_PITCH_P, MC_YAW_P

    if profile == "simulation":
        MPC_XY_P = 0.30
        MPC_Z_P = 1.00
        MPC_XY_VEL_P_ACC = 1.80
        MPC_XY_VEL_I_ACC = 0.40
        MPC_XY_VEL_D_ACC = 0.20
        MPC_Z_VEL_P_ACC = 4.00
        MPC_Z_VEL_I_ACC = 2.00
        MPC_Z_VEL_D_ACC = 0.00
        MC_ROLL_P = 4.00
        MC_PITCH_P = 4.00
        MC_YAW_P = 2.80
    elif profile == "current":
        MPC_XY_P = 0.95
        MPC_Z_P = 1.00
        MPC_XY_VEL_P_ACC = 1.80
        MPC_XY_VEL_I_ACC = 0.40
        MPC_XY_VEL_D_ACC = 0.20
        MPC_Z_VEL_P_ACC = 4.00
        MPC_Z_VEL_I_ACC = 2.00
        MPC_Z_VEL_D_ACC = 0.00
        MC_ROLL_P = 4.00
        MC_PITCH_P = 6.80
        MC_YAW_P = 2.80
    elif profile == "legacy":
        MPC_XY_P = 0.30
        MPC_Z_P = 1.00
        MPC_XY_VEL_P_ACC = 1.80
        MPC_XY_VEL_I_ACC = 0.00
        MPC_XY_VEL_D_ACC = 0.20
        MPC_Z_VEL_P_ACC = 4.00
        MPC_Z_VEL_I_ACC = 2.00
        MPC_Z_VEL_D_ACC = 0.00
        MC_ROLL_P = 4.00
        MC_PITCH_P = 2.00
        MC_YAW_P = 2.80
    else:  # px4
        MPC_XY_P = 0.95
        MPC_Z_P = 1.00
        MPC_XY_VEL_P_ACC = 1.80
        MPC_XY_VEL_I_ACC = 0.40
        MPC_XY_VEL_D_ACC = 0.20
        MPC_Z_VEL_P_ACC = 4.00
        MPC_Z_VEL_I_ACC = 2.00
        MPC_Z_VEL_D_ACC = 0.00
        MC_ROLL_P = 4.00
        MC_PITCH_P = 2.00
        MC_YAW_P = 2.80


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def quat_to_yaw(q0, q1, q2, q3):
    return np.arctan2(
        2.0 * (q0 * q3 + q1 * q2),
        q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    )


def quat_mul(q, r):
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    return np.array([
        q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3,
        q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2,
        q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1,
        q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0,
    ])


def integrate_quat(q, omega, step):
    # q_dot = 0.5 * q ⊗ [0, wx, wy, wz]
    q_dot = 0.5 * quat_mul(q, np.array([0.0, omega[0], omega[1], omega[2]]))
    return normalize(q + q_dot * step)


def limit_tilt(body_z, max_tilt):
    world_z = np.array([0.0, 0.0, 1.0])
    body_z = normalize(body_z)
    dot = np.clip(np.dot(body_z, world_z), -1.0, 1.0)
    angle = np.arccos(dot)
    if angle <= max_tilt:
        return body_z

    rejection = body_z - dot * world_z
    if np.dot(rejection, rejection) < 1e-8:
        rejection = np.array([1.0, 0.0, 0.0])
    rejection = normalize(rejection)
    return np.cos(max_tilt) * world_z + np.sin(max_tilt) * rejection


def constrain_xy(v0, v1, max_norm):
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)

    if np.linalg.norm(v0 + v1) <= max_norm:
        return v0 + v1
    if np.linalg.norm(v0) >= max_norm:
        return normalize(v0) * max_norm
    if np.linalg.norm(v1 - v0) < 1e-3:
        return normalize(v0) * max_norm
    if np.linalg.norm(v0) < 1e-3:
        return normalize(v1) * max_norm

    u1 = normalize(v1)
    m = np.dot(u1, v0)
    c = np.dot(v0, v0) - max_norm * max_norm
    s = -m + np.sqrt(max(0.0, m * m - c))
    return v0 + u1 * s


def rotmat_to_quat(body_x, body_y, body_z):
    r00, r01, r02 = body_x[0], body_y[0], body_z[0]
    r10, r11, r12 = body_x[1], body_y[1], body_z[1]
    r20, r21, r22 = body_x[2], body_y[2], body_z[2]
    tr = r00 + r11 + r22

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * s
        q1 = (r21 - r12) / s
        q2 = (r02 - r20) / s
        q3 = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = np.sqrt(1.0 + r00 - r11 - r22) * 2.0
        q0 = (r21 - r12) / s
        q1 = 0.25 * s
        q2 = (r01 + r10) / s
        q3 = (r02 + r20) / s
    elif r11 > r22:
        s = np.sqrt(1.0 + r11 - r00 - r22) * 2.0
        q0 = (r02 - r20) / s
        q1 = (r01 + r10) / s
        q2 = 0.25 * s
        q3 = (r12 + r21) / s
    else:
        s = np.sqrt(1.0 + r22 - r00 - r11) * 2.0
        q0 = (r10 - r01) / s
        q1 = (r02 + r20) / s
        q2 = (r12 + r21) / s
        q3 = 0.25 * s

    return normalize(np.array([q0, q1, q2, q3]))


def thrust_to_attitude(thr_sp, yaw_sp):
    body_z = -thr_sp.copy()
    if np.dot(body_z, body_z) < 1e-8:
        body_z = np.array([0.0, 0.0, 1.0])
    body_z = normalize(body_z)

    y_c = np.array([-np.sin(yaw_sp), np.cos(yaw_sp), 0.0])
    body_x = np.cross(y_c, body_z)

    if body_z[2] < 0.0:
        body_x = -body_x
    if abs(body_z[2]) < 1e-6:
        body_x = np.array([0.0, 0.0, 1.0])

    body_x = normalize(body_x)
    body_y = np.cross(body_z, body_x)
    return rotmat_to_quat(body_x, body_y, body_z)


def sin_cos_simulator(x):
    if x > np.pi:
        x -= 2.0 * np.pi
    if x < -np.pi:
        x += 2.0 * np.pi
    x2 = x * x
    s = x * (1.0 - x2 / 6.0 + x2 * x2 / 120.0)
    c = 1.0 - x2 / 2.0 + x2 * x2 / 24.0
    return s, c


def thrust_to_attitude_simulator(thr_sp, yaw_sp):
    body_z = -thr_sp.copy()
    n = np.linalg.norm(body_z)
    if n < 1e-8:
        body_z = np.array([0.0, 0.0, 1.0])
        n = 1.0
    body_z = body_z / n

    sy, cy = sin_cos_simulator(yaw_sp)
    y_c = np.array([-sy, cy, 0.0])
    body_x = np.cross(y_c, body_z)

    if body_z[2] < 0.0:
        body_x = -body_x
    if abs(body_z[2]) < 1e-6:
        body_x = np.array([0.0, 0.0, 1.0])

    body_x = normalize(body_x)
    body_y = np.cross(body_z, body_x)
    return rotmat_to_quat(body_x, body_y, body_z)


def align_quat_sign(q, reference):
    q = np.asarray(q, dtype=float).copy()
    reference = np.asarray(reference, dtype=float)
    if np.all(np.isfinite(q)) and np.all(np.isfinite(reference)) and np.dot(q, reference) < 0.0:
        q = -q
    return q


def get_yaw_setpoint_from_row(row):
    if "u0_yaw" in row:
        return row["u0_yaw"]
    if {"target_e0", "target_e1", "target_e2", "target_e3"}.issubset(row.index):
        return quat_to_yaw(row["target_e0"], row["target_e1"], row["target_e2"], row["target_e3"])
    return quat_to_yaw(row["cur_e0"], row["cur_e1"], row["cur_e2"], row["cur_e3"])


def px4_nominal_setpoints(
    pos,
    vel,
    q,
    pos_sp,
    yaw_sp,
    prev_vel,
    vel_dot_lpf,
    vel_int,
    step_dt,
    vel_sp_override=None,
    acc_sp_override=None,
    att_sp_override=None,
    attitude_func=thrust_to_attitude,
):
    # ==============================
    # PX4 _positionControl(): nominal velocity setpoint before optional RLS correction.
    # ==============================
    vel_sp_position = np.array([
        MPC_XY_P * (pos_sp[0] - pos[0]),
        MPC_XY_P * (pos_sp[1] - pos[1]),
        MPC_Z_P * (pos_sp[2] - pos[2]),
    ])

    vel_sp_nominal = vel_sp_position.copy()
    vel_sp_nominal[:2] = constrain_xy(
        vel_sp_position[:2],
        vel_sp_nominal[:2] - vel_sp_position[:2],
        MPC_XY_VEL_MAX,
    )
    vel_sp_nominal[2] = np.clip(vel_sp_nominal[2], -MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DOWN)
    vel_sp = vel_sp_nominal.copy() if vel_sp_override is None else np.asarray(vel_sp_override, dtype=float).copy()

    # ==============================
    # PX4 _velocityControl(): nominal acceleration setpoint before optional RLS correction.
    # ==============================
    vel_error = vel_sp - vel

    if MPC_VELD_LP > 1e-6:
        vel_dot_alpha = step_dt / (step_dt + 1.0 / (2.0 * np.pi * MPC_VELD_LP))
    else:
        vel_dot_alpha = 1.0

    vel_dot_raw = (vel - prev_vel) / step_dt
    vel_dot_lpf = vel_dot_lpf + vel_dot_alpha * (vel_dot_raw - vel_dot_lpf)
    vel_dot = vel_dot_lpf

    acc_sp_nominal = np.zeros(3)
    acc_sp_nominal[0] = MPC_XY_VEL_P_ACC * vel_error[0] + vel_int[0] - MPC_XY_VEL_D_ACC * vel_dot[0]
    acc_sp_nominal[1] = MPC_XY_VEL_P_ACC * vel_error[1] + vel_int[1] - MPC_XY_VEL_D_ACC * vel_dot[1]
    acc_sp_nominal[2] = MPC_Z_VEL_P_ACC * vel_error[2] + vel_int[2] - MPC_Z_VEL_D_ACC * vel_dot[2]
    acc_sp = acc_sp_nominal.copy() if acc_sp_override is None else np.asarray(acc_sp_override, dtype=float).copy()

    # ==============================
    # PX4 _accelerationControl(): nominal attitude setpoint before optional RLS correction.
    # ==============================
    z_specific_force = -A_OF_GRAVITY
    body_z = np.array([-acc_sp[0], -acc_sp[1], -z_specific_force])
    body_z = normalize(body_z)
    body_z = limit_tilt(body_z, MPC_TILT_MAX)

    thrust_ned_z = acc_sp[2] * (MPC_THR_HOVER / A_OF_GRAVITY) - MPC_THR_HOVER
    cos_ned_body = np.dot(np.array([0.0, 0.0, 1.0]), body_z)
    if abs(cos_ned_body) < 1e-6:
        cos_ned_body = 1e-6

    collective_thrust = min(thrust_ned_z / cos_ned_body, -MPC_THR_MIN)
    thr_sp = body_z * collective_thrust

    thrust_sp_xy_norm = np.linalg.norm(thr_sp[:2])
    thrust_max_squared = MPC_THR_MAX * MPC_THR_MAX
    allocated_horizontal_thrust = min(thrust_sp_xy_norm, MPC_THR_XY_MARGIN)
    thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust ** 2
    thr_sp[2] = max(thr_sp[2], -np.sqrt(max(0.0, thrust_z_max_squared)))

    thrust_max_xy_squared = thrust_max_squared - thr_sp[2] ** 2
    thrust_max_xy = np.sqrt(max(0.0, thrust_max_xy_squared))
    if thrust_sp_xy_norm > thrust_max_xy and thrust_sp_xy_norm > 1e-8:
        thr_sp[:2] = thr_sp[:2] / thrust_sp_xy_norm * thrust_max_xy

    # Z anti-windup only. XY ARW is intentionally disabled, same as setpoint viewer.
    vel_error_for_int = vel_error.copy()
    if (thr_sp[2] >= -MPC_THR_MIN and vel_error_for_int[2] >= 0.0) or \
       (thr_sp[2] <= -MPC_THR_MAX and vel_error_for_int[2] <= 0.0):
        vel_error_for_int[2] = 0.0

    vel_error_for_int = np.nan_to_num(vel_error_for_int)
    vel_int_next = vel_int.copy()
    vel_int_next[0] += vel_error_for_int[0] * MPC_XY_VEL_I_ACC * step_dt
    vel_int_next[1] += vel_error_for_int[1] * MPC_XY_VEL_I_ACC * step_dt
    vel_int_next[2] += vel_error_for_int[2] * MPC_Z_VEL_I_ACC * step_dt
    vel_int_next[2] = np.clip(vel_int_next[2], -A_OF_GRAVITY, A_OF_GRAVITY)

    att_sp_nominal = attitude_func(thr_sp, yaw_sp)
    att_sp = att_sp_nominal.copy() if att_sp_override is None else normalize(att_sp_override)

    # ==============================
    # attitude setpoint -> nominal rate setpoint
    # ==============================
    qe0 = q[0] * att_sp[0] + q[1] * att_sp[1] + q[2] * att_sp[2] + q[3] * att_sp[3]
    sgn = 1.0 if qe0 >= 0.0 else -1.0

    rate_nominal = np.zeros(3)
    rate_nominal[0] = 2.0 * MC_ROLL_P * sgn * (
        q[0] * att_sp[1] - q[1] * att_sp[0] - q[2] * att_sp[3] + q[3] * att_sp[2]
    )
    rate_nominal[1] = 2.0 * MC_PITCH_P * sgn * (
        q[0] * att_sp[2] + q[1] * att_sp[3] - q[2] * att_sp[0] - q[3] * att_sp[1]
    )
    rate_nominal[2] = 2.0 * MC_YAW_P * sgn * (
        q[0] * att_sp[3] - q[1] * att_sp[2] + q[2] * att_sp[1] - q[3] * att_sp[0]
    )

    return {
        "vel_sp": vel_sp,
        "vel_sp_nominal": vel_sp_nominal,
        "acc_sp": acc_sp,
        "acc_sp_nominal": acc_sp_nominal,
        "att_sp": att_sp,
        "att_sp_nominal": att_sp_nominal,
        "rate_nominal": rate_nominal,
        "thr_sp": thr_sp,
        "vel_error": vel_error,
        "vel_dot": vel_dot,
        "vel_dot_lpf": vel_dot_lpf,
        "vel_int_next": vel_int_next,
    }


def simulator_setpoints(pos, vel, q, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, step_dt):
    return px4_nominal_setpoints(
        pos,
        vel,
        q,
        pos_sp,
        yaw_sp,
        prev_vel,
        vel_dot_lpf,
        vel_int,
        step_dt,
        attitude_func=thrust_to_attitude_simulator,
    )


def simulator_state_step(s, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, step_dt):
    q_prev = normalize(s[0:4])
    if np.linalg.norm(q_prev) < 1e-8:
        q_prev = np.array([1.0, 0.0, 0.0, 0.0])

    pos = s[[state_index["x"], state_index["y"], state_index["z"]]].copy()
    vel_prev = s[[state_index["vx"], state_index["vy"], state_index["vz"]]].copy()
    w_prev = s[[state_index["wx"], state_index["wy"], state_index["wz"]]].copy()

    nominal = simulator_setpoints(pos, vel_prev, q_prev, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, step_dt)
    acc_sp = nominal["acc_sp"]
    omega_sp = nominal["rate_nominal"]

    s_next = s.copy()
    s_next[state_index["x"]] += MODEL_POS_ALPHA[0] * vel_prev[0] * step_dt
    s_next[state_index["y"]] += MODEL_POS_ALPHA[1] * vel_prev[1] * step_dt
    s_next[state_index["z"]] += MODEL_POS_ALPHA[2] * vel_prev[2] * step_dt

    vel_next = MODEL_VEL_A * vel_prev + MODEL_VEL_B * acc_sp
    s_next[state_index["vx"]] = vel_next[0]
    s_next[state_index["vy"]] = vel_next[1]
    s_next[state_index["vz"]] = vel_next[2]

    q_dot = 0.5 * quat_mul(q_prev, np.array([0.0, w_prev[0], w_prev[1], w_prev[2]]))
    s_next[0:4] = normalize(q_prev + MODEL_ATT_BETA * q_dot * step_dt)

    w_next = MODEL_W_C * w_prev + MODEL_W_D * omega_sp
    s_next[state_index["wx"]] = w_next[0]
    s_next[state_index["wy"]] = w_next[1]
    s_next[state_index["wz"]] = w_next[2]

    return s_next, omega_sp, nominal["vel_dot_lpf"], nominal["vel_int_next"]


def simulator_state_step_with_setpoint_rls(
    s,
    pos_sp,
    yaw_sp,
    prev_vel,
    vel_dot_lpf,
    vel_int,
    step_dt,
    ctx=None,
):
    q_prev = normalize(s[0:4])
    if np.linalg.norm(q_prev) < 1e-8:
        q_prev = np.array([1.0, 0.0, 0.0, 0.0])

    pos = s[[state_index["x"], state_index["y"], state_index["z"]]].copy()
    vel_prev = s[[state_index["vx"], state_index["vy"], state_index["vz"]]].copy()
    w_prev = s[[state_index["wx"], state_index["wy"], state_index["wz"]]].copy()

    if ctx is None:
        nominal = simulator_setpoints(pos, vel_prev, q_prev, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, step_dt)
        omega_sp = nominal["rate_nominal"]
    else:
        rls_theta = {
            "vel_sp": ctx["vel_sp_theta"],
            "acc_sp": ctx["acc_sp_theta"],
            "att_sp": ctx["att_sp_theta"],
        }
        rls_errors = {
            "vel_sp": ctx["prev_vel_sp_error"],
            "acc_sp": ctx["prev_acc_sp_error"],
            "att_sp": ctx["prev_att_sp_error"],
        }
        nominal = corrected_px4_setpoints(
            pos,
            vel_prev,
            q_prev,
            pos_sp,
            yaw_sp,
            prev_vel,
            vel_dot_lpf,
            vel_int,
            step_dt,
            rls_theta,
            rls_errors,
            "online",
            attitude_func=thrust_to_attitude_simulator,
        )
        omega_sp = apply_rate_rls(
            nominal["rate_nominal"],
            ctx["rate_theta"],
            ctx["prev_rate_sp_error"],
            "online",
        )
        ctx["prev_vel_sp_error"] = np.zeros(3)
        ctx["prev_acc_sp_error"] = np.zeros(3)
        ctx["prev_att_sp_error"] = np.zeros(4)
        ctx["prev_rate_sp_error"] = np.zeros(3)

    acc_sp = nominal["acc_sp"]

    s_next = s.copy()
    s_next[state_index["x"]] += MODEL_POS_ALPHA[0] * vel_prev[0] * step_dt
    s_next[state_index["y"]] += MODEL_POS_ALPHA[1] * vel_prev[1] * step_dt
    s_next[state_index["z"]] += MODEL_POS_ALPHA[2] * vel_prev[2] * step_dt

    vel_next = MODEL_VEL_A * vel_prev + MODEL_VEL_B * acc_sp
    s_next[state_index["vx"]] = vel_next[0]
    s_next[state_index["vy"]] = vel_next[1]
    s_next[state_index["vz"]] = vel_next[2]

    q_dot = 0.5 * quat_mul(q_prev, np.array([0.0, w_prev[0], w_prev[1], w_prev[2]]))
    s_next[0:4] = normalize(q_prev + MODEL_ATT_BETA * q_dot * step_dt)

    w_next = MODEL_W_C * w_prev + MODEL_W_D * omega_sp
    s_next[state_index["wx"]] = w_next[0]
    s_next[state_index["wy"]] = w_next[1]
    s_next[state_index["wz"]] = w_next[2]

    return s_next, omega_sp, acc_sp, nominal["vel_dot_lpf"], nominal["vel_int_next"]


def get_position_setpoint_from_row(row, h=0):
    x_ref = row[f"u{h}_x"] if f"u{h}_x" in row.index else row.get("u0_x", row.get("target_x", row.get("cur_x", 0.0)))
    y_ref = row[f"u{h}_y"] if f"u{h}_y" in row.index else row.get("u0_y", row.get("target_y", row.get("cur_y", 0.0)))
    z_ref = row[f"u{h}_z"] if f"u{h}_z" in row.index else row.get("u0_z", row.get("target_z", row.get("cur_z", 0.0)))
    return np.array([x_ref, y_ref, z_ref], dtype=float)


def get_yaw_setpoint_from_row_horizon(row, h=0):
    if f"u{h}_yaw" in row.index:
        return row[f"u{h}_yaw"]
    return get_yaw_setpoint_from_row(row)


def nearest_square_waypoint_index(target):
    target = np.asarray(target, dtype=float)
    d = np.linalg.norm(SQUARE_WAYPOINTS[:, :2] - target[:2], axis=1)
    idx = int(np.argmin(d))
    if d[idx] < 0.25:
        return idx
    if np.linalg.norm(target[:2] - SQUARE_START_TARGET[:2]) < 0.25:
        return -1
    return idx


def square_target_from_index(index):
    if index < 0:
        return SQUARE_START_TARGET.copy()
    return SQUARE_WAYPOINTS[min(index, len(SQUARE_WAYPOINTS) - 1)].copy()


def build_square_target_history(df):
    n = len(df)
    target_index_hist = np.full(n, -1, dtype=int)
    target_change_time_hist = np.zeros(n, dtype=float)

    if not {"target_x", "target_y", "target_z"}.issubset(df.columns):
        return target_index_hist, target_change_time_hist

    last_idx = None
    last_change_time = float(df.iloc[0]["t"])
    for i, row in df.iterrows():
        target = row[["target_x", "target_y", "target_z"]].to_numpy(float)
        idx = nearest_square_waypoint_index(target)
        if last_idx is None or idx != last_idx:
            last_idx = idx
            last_change_time = float(row["t"])
        target_index_hist[i] = idx
        target_change_time_hist[i] = last_change_time

    return target_index_hist, target_change_time_hist


def maybe_advance_square_target(route, pos, current_time):
    if route["index"] >= len(SQUARE_WAYPOINTS) - 1:
        return

    target = square_target_from_index(route["index"])
    xy_error = np.linalg.norm(target[:2] - pos[:2])
    elapsed = current_time - route["change_time"]
    if elapsed >= SQUARE_WAYPOINT_HOLD_SEC and xy_error < SQUARE_WAYPOINT_THRESHOLD:
        route["index"] += 1
        route["change_time"] = current_time


def get_prediction_position_setpoint(row, h, route, s, current_time):
    maybe_advance_square_target(route, s[[state_index["x"], state_index["y"], state_index["z"]]], current_time)

    if route["index"] > route["row_index"]:
        return square_target_from_index(route["index"])

    return get_position_setpoint_from_row(row, h)


def get_prediction_yaw_setpoint(row, h, route):
    if route["index"] > route["row_index"]:
        return 0.0
    return get_yaw_setpoint_from_row_horizon(row, h)


def build_simulator_history(df):
    n = len(df)
    vel_dot_lpf_hist = np.zeros((n, 3), dtype=float)
    vel_int_hist = np.zeros((n, 3), dtype=float)
    omega_sp_hist = np.zeros((n, 3), dtype=float)
    internal_w_next_hist = np.zeros((n, 3), dtype=float)

    prev_vel = df.loc[0, ["cur_vx", "cur_vy", "cur_vz"]].to_numpy(float)
    vel_dot_lpf = np.zeros(3)
    vel_int = np.zeros(3)

    for i in range(n):
        row = df.iloc[i]
        s = np.zeros(len(state_names))
        for name in state_names:
            col = f"cur_{name}"
            if col in df.columns:
                s[state_index[name]] = row[col]
        s[0:4] = normalize(s[0:4])
        if np.linalg.norm(s[0:4]) < 1e-8:
            s[0:4] = np.array([1.0, 0.0, 0.0, 0.0])

        pos_sp = get_position_setpoint_from_row(row, 0)
        yaw_sp = get_yaw_setpoint_from_row_horizon(row, 0)

        vel_dot_lpf_hist[i] = vel_dot_lpf
        vel_int_hist[i] = vel_int
        s_next, omega_sp, vel_dot_lpf, vel_int = simulator_state_step(
            s, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, dt
        )
        omega_sp_hist[i] = omega_sp
        internal_w_next_hist[i] = s_next[[state_index["wx"], state_index["wy"], state_index["wz"]]]
        prev_vel = s_next[[state_index["vx"], state_index["vy"], state_index["vz"]]].copy()

    return vel_dot_lpf_hist, vel_int_hist, omega_sp_hist, internal_w_next_hist


def make_setpoint_rls_models():
    vel_sp_models = [
        RLSModel(
            [MPC_XY_P if axis < 2 else MPC_Z_P, 0.0, 0.0],
            lower=np.array([-20.0, -0.99, -10.0]),
            upper=np.array([20.0, 0.999, 10.0]),
        )
        for axis in range(3)
    ]
    acc_sp_models = [
        RLSModel(
            [
                MPC_XY_VEL_P_ACC if axis < 2 else MPC_Z_VEL_P_ACC,
                1.0,
                MPC_XY_VEL_D_ACC if axis < 2 else MPC_Z_VEL_D_ACC,
                0.0,
                0.0,
            ],
            lower=np.array([-20.0, -20.0, -20.0, -0.99, -20.0]),
            upper=np.array([20.0, 20.0, 20.0, 0.999, 20.0]),
        )
        for axis in range(3)
    ]
    att_sp_models = [
        RLSModel(
            [1.0, 0.0, 0.0],
            lower=np.array([-2.0, -0.99, -2.0]),
            upper=np.array([2.0, 0.999, 2.0]),
        )
        for _ in range(4)
    ]
    rate_sp_models = [
        RLSModel(
            [1.0, 0.0, 0.0],
            lower=np.array([-20.0, -0.99, -10.0]),
            upper=np.array([20.0, 0.999, 10.0]),
        )
        for _ in range(3)
    ]
    return vel_sp_models, acc_sp_models, att_sp_models, rate_sp_models


def apply_velocity_sp_rls(pos, pos_sp, vel_sp_nominal, theta, prev_vel_sp_error, id_mode):
    if id_mode != "online":
        return vel_sp_nominal.copy()

    pos_err = pos_sp - pos
    corrected = np.zeros(3)
    for axis in range(3):
        phi = np.array([pos_err[axis], prev_vel_sp_error[axis], 1.0])
        y_hat = phi @ theta[axis]
        corrected[axis] = vel_sp_nominal[axis] if not np.isfinite(y_hat) else y_hat
    return corrected


def apply_acc_sp_rls(vel_error, vel_int, vel_dot, acc_sp_nominal, theta, prev_acc_sp_error, id_mode):
    if id_mode != "online":
        return acc_sp_nominal.copy()

    corrected = np.zeros(3)
    for axis in range(3):
        phi = np.array([
            vel_error[axis],
            vel_int[axis],
            -vel_dot[axis],
            prev_acc_sp_error[axis],
            1.0,
        ])
        y_hat = phi @ theta[axis]
        corrected[axis] = acc_sp_nominal[axis] if not np.isfinite(y_hat) else y_hat
    return corrected


def apply_att_sp_rls(att_sp_nominal, theta, prev_att_sp_error, id_mode):
    if id_mode != "online":
        return att_sp_nominal.copy()

    corrected = np.zeros(4)
    for axis in range(4):
        phi = np.array([att_sp_nominal[axis], prev_att_sp_error[axis], 1.0])
        y_hat = phi @ theta[axis]
        corrected[axis] = att_sp_nominal[axis] if not np.isfinite(y_hat) else y_hat
    corrected = normalize(corrected)
    return align_quat_sign(corrected, att_sp_nominal)


# PX4のnominal setpoint計算に対して、RLSで補正したvel_sp, acc_sp, att_spを反映させたsetpointを再計算
def corrected_px4_setpoints(
    pos,
    vel,
    q,
    pos_sp,
    yaw_sp,
    prev_vel,
    vel_dot_lpf,
    vel_int,
    step_dt,
    rls_theta,
    rls_errors,
    id_mode,
    attitude_func=thrust_to_attitude,
):
    nominal = px4_nominal_setpoints(
        pos, vel, q, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, step_dt, attitude_func=attitude_func
    )
    vel_sp = apply_velocity_sp_rls(
        pos, pos_sp, nominal["vel_sp_nominal"], rls_theta["vel_sp"], rls_errors["vel_sp"], id_mode
    )
    # RLS補正したvel_spからacc_sp, att_spを再計算
    with_vel = px4_nominal_setpoints(
        pos,
        vel,
        q,
        pos_sp,
        yaw_sp,
        prev_vel,
        vel_dot_lpf,
        vel_int,
        step_dt,
        vel_sp_override=vel_sp,
        attitude_func=attitude_func,
    )
    acc_sp = apply_acc_sp_rls(
        with_vel["vel_error"],
        vel_int,
        with_vel["vel_dot"],
        with_vel["acc_sp_nominal"],
        rls_theta["acc_sp"],
        rls_errors["acc_sp"],
        id_mode,
    )
    # RLS補正したacc_spからatt_spを再計算
    with_acc = px4_nominal_setpoints(
        pos,
        vel,
        q,
        pos_sp,
        yaw_sp,
        prev_vel,
        vel_dot_lpf,
        vel_int,
        step_dt,
        vel_sp_override=vel_sp,
        acc_sp_override=acc_sp,
        attitude_func=attitude_func,
    )
    att_sp = apply_att_sp_rls(with_acc["att_sp_nominal"], rls_theta["att_sp"], rls_errors["att_sp"], id_mode)

    return px4_nominal_setpoints(
        pos,
        vel,
        q,
        pos_sp,
        yaw_sp,
        prev_vel,
        vel_dot_lpf,
        vel_int,
        step_dt,
        vel_sp_override=vel_sp,
        acc_sp_override=acc_sp,
        att_sp_override=att_sp,
        attitude_func=attitude_func,
    )


def build_rls_history(df, id_mode):
    n = len(df)

    theta_hist = np.zeros((n, 3, 3), dtype=float)
    vel_sp_theta_hist = np.zeros((n, 3, 3), dtype=float)
    acc_sp_theta_hist = np.zeros((n, 3, 5), dtype=float)
    att_sp_theta_hist = np.zeros((n, 4, 3), dtype=float)
    vel_state_theta_hist = np.zeros((n, 3, 4), dtype=float)
    prev_vel_sp_error_hist = np.zeros((n, 3), dtype=float)
    prev_acc_sp_error_hist = np.zeros((n, 3), dtype=float)
    prev_att_sp_error_hist = np.zeros((n, 4), dtype=float)
    prev_rate_sp_error_hist = np.zeros((n, 3), dtype=float)
    prev_vel_state_error_hist = np.zeros((n, 3), dtype=float)
    nominal_rate_hist = np.zeros((n, 3), dtype=float)
    corrected_rate_hist = np.zeros((n, 3), dtype=float)
    corrected_acc_sp_hist = np.zeros((n, 3), dtype=float)
    vel_dot_lpf_hist = np.zeros((n, 3), dtype=float)
    vel_int_hist = np.zeros((n, 3), dtype=float)

    vel_sp_models, acc_sp_models, att_sp_models, rate_sp_models = make_setpoint_rls_models()
    vel_state_models = [
        RLSModel(
            [1.0, dt, 0.0, 0.0],
            lower=np.array([-2.0, -2.0, -10.0, -10.0]),
            upper=np.array([2.0, 2.0, 10.0, 10.0]),
            p_scale=100.0,
            forgetting=0.995,
        )
        for _ in range(3)
    ]

    prev_vel = df.loc[0, ["cur_vx", "cur_vy", "cur_vz"]].to_numpy(float)
    vel_dot_lpf = np.zeros(3)
    vel_int = np.zeros(3)
    prev_observed_vel_sp = np.zeros(3)
    prev_modeled_vel_sp = np.zeros(3)
    prev_observed_acc_sp = np.zeros(3)
    prev_modeled_acc_sp = np.zeros(3)
    prev_observed_att_sp = np.array([1.0, 0.0, 0.0, 0.0])
    prev_modeled_att_sp = np.array([1.0, 0.0, 0.0, 0.0])
    prev_observed_rate_sp = np.zeros(3)
    prev_modeled_rate_sp = np.zeros(3)
    prev_observed_vel_state = prev_vel.copy()
    prev_modeled_vel_state = prev_vel.copy()

    for i in range(n):
        row = df.iloc[i]

        pos = row[["cur_x", "cur_y", "cur_z"]].to_numpy(float)
        vel = row[["cur_vx", "cur_vy", "cur_vz"]].to_numpy(float)
        q = normalize(row[["cur_e0", "cur_e1", "cur_e2", "cur_e3"]].to_numpy(float))

        if {"u0_x", "u0_y", "u0_z"}.issubset(df.columns):
            pos_sp = row[["u0_x", "u0_y", "u0_z"]].to_numpy(float)
        yaw_sp = get_yaw_setpoint_from_row(row)

        vel_dot_lpf_hist[i] = vel_dot_lpf
        vel_int_hist[i] = vel_int
        prev_vel_sp_error = prev_observed_vel_sp - prev_modeled_vel_sp
        prev_acc_sp_error = prev_observed_acc_sp - prev_modeled_acc_sp
        prev_att_sp_error = prev_observed_att_sp - prev_modeled_att_sp
        prev_rate_sp_error = prev_observed_rate_sp - prev_modeled_rate_sp
        prev_vel_state_error = prev_observed_vel_state - prev_modeled_vel_state
        prev_vel_sp_error_hist[i] = prev_vel_sp_error
        prev_acc_sp_error_hist[i] = prev_acc_sp_error
        prev_att_sp_error_hist[i] = prev_att_sp_error
        prev_rate_sp_error_hist[i] = prev_rate_sp_error
        prev_vel_state_error_hist[i] = prev_vel_state_error

        rls_theta = {
            "vel_sp": np.array([model.theta for model in vel_sp_models]),
            "acc_sp": np.array([model.theta for model in acc_sp_models]),
            "att_sp": np.array([model.theta for model in att_sp_models]),
        }
        rls_errors = {
            "vel_sp": prev_vel_sp_error,
            "acc_sp": prev_acc_sp_error,
            "att_sp": prev_att_sp_error,
        }
        # RLS補正したsetpoint
        nominal = corrected_px4_setpoints(
            pos, vel, q, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, dt, rls_theta, rls_errors, id_mode
        )
        corrected_acc_sp_hist[i] = nominal["acc_sp"]
        rate_nominal = nominal["rate_nominal"]
        nominal_rate_hist[i] = rate_nominal

        corrected_rate = rate_nominal.copy()
        if id_mode == "online":
            for axis in range(3):
                phi = np.array([rate_nominal[axis], prev_rate_sp_error[axis], 1.0])
                y_hat = rate_sp_models[axis].predict(phi)
                corrected_rate[axis] = rate_nominal[axis] if not np.isfinite(y_hat) else y_hat

        # RLS補正したrate_sp
        corrected_rate_hist[i] = corrected_rate
        for axis in range(3):
            theta_hist[i, axis] = rate_sp_models[axis].theta
            vel_sp_theta_hist[i, axis] = vel_sp_models[axis].theta
            acc_sp_theta_hist[i, axis] = acc_sp_models[axis].theta
            vel_state_theta_hist[i, axis] = vel_state_models[axis].theta
        for axis in range(4):
            att_sp_theta_hist[i, axis] = att_sp_models[axis].theta

        modeled_next_vel = np.zeros(3)
        if id_mode == "online":
            for axis in range(3):
                phi = np.array([vel[axis], nominal["acc_sp"][axis], prev_vel_state_error[axis], 1.0])
                y_hat = vel_state_models[axis].predict(phi)
                modeled_next_vel[axis] = vel[axis] + nominal["acc_sp"][axis] * dt if not np.isfinite(y_hat) else y_hat
        else:
            modeled_next_vel = vel + nominal["acc_sp"] * dt

        vel_dot_lpf = nominal["vel_dot_lpf"]
        vel_int = nominal["vel_int_next"]

        if id_mode == "online" and {"px4_sp_vx", "px4_sp_vy", "px4_sp_vz"}.issubset(df.columns):
            logged_vel_sp = row[["px4_sp_vx", "px4_sp_vy", "px4_sp_vz"]].to_numpy(float)
            pos_err = pos_sp - pos
            for axis in range(3):
                phi = np.array([pos_err[axis], prev_vel_sp_error[axis], 1.0])
                vel_sp_models[axis].update(phi, logged_vel_sp[axis])
            if np.all(np.isfinite(logged_vel_sp)):
                prev_observed_vel_sp = logged_vel_sp.copy()
            else:
                prev_observed_vel_sp = nominal["vel_sp"].copy()
        else:
            prev_observed_vel_sp = nominal["vel_sp"].copy()
        prev_modeled_vel_sp = nominal["vel_sp"].copy()

        if id_mode == "online" and {"px4_sp_acc_x", "px4_sp_acc_y", "px4_sp_acc_z"}.issubset(df.columns):
            logged_acc_sp = row[["px4_sp_acc_x", "px4_sp_acc_y", "px4_sp_acc_z"]].to_numpy(float)
            for axis in range(3):
                phi = np.array([
                    nominal["vel_error"][axis],
                    vel_int_hist[i, axis],
                    -nominal["vel_dot"][axis],
                    prev_acc_sp_error[axis],
                    1.0,
                ])
                acc_sp_models[axis].update(phi, logged_acc_sp[axis])
            if np.all(np.isfinite(logged_acc_sp)):
                prev_observed_acc_sp = logged_acc_sp.copy()
            else:
                prev_observed_acc_sp = nominal["acc_sp"].copy()
        else:
            prev_observed_acc_sp = nominal["acc_sp"].copy()
        prev_modeled_acc_sp = nominal["acc_sp"].copy()

        if id_mode == "online" and {"px4_sp_q0", "px4_sp_q1", "px4_sp_q2", "px4_sp_q3"}.issubset(df.columns):
            logged_att_sp = row[["px4_sp_q0", "px4_sp_q1", "px4_sp_q2", "px4_sp_q3"]].to_numpy(float)
            logged_att_sp = align_quat_sign(logged_att_sp, nominal["att_sp"])
            for axis in range(4):
                phi = np.array([nominal["att_sp_nominal"][axis], prev_att_sp_error[axis], 1.0])
                att_sp_models[axis].update(phi, logged_att_sp[axis])
            if np.all(np.isfinite(logged_att_sp)):
                prev_observed_att_sp = logged_att_sp.copy()
            else:
                prev_observed_att_sp = nominal["att_sp"].copy()
        else:
            prev_observed_att_sp = nominal["att_sp"].copy()
        prev_modeled_att_sp = nominal["att_sp"].copy()

        if id_mode == "online" and all(col in df.columns for col in rate_cols):
            logged_rate_sp = row[rate_cols].to_numpy(float)
            for axis in range(3):
                phi = np.array([rate_nominal[axis], prev_rate_sp_error[axis], 1.0])
                rate_sp_models[axis].update(phi, logged_rate_sp[axis])
            if np.all(np.isfinite(logged_rate_sp)):
                prev_observed_rate_sp = logged_rate_sp.copy()
            else:
                prev_observed_rate_sp = corrected_rate.copy()
        else:
            prev_observed_rate_sp = corrected_rate.copy()

        if id_mode == "online" and i + 1 < n:
            logged_next_vel = df.iloc[i + 1][["cur_vx", "cur_vy", "cur_vz"]].to_numpy(float)
            for axis in range(3):
                phi = np.array([vel[axis], nominal["acc_sp"][axis], prev_vel_state_error[axis], 1.0])
                vel_state_models[axis].update(phi, logged_next_vel[axis])
            if np.all(np.isfinite(logged_next_vel)):
                prev_observed_vel_state = logged_next_vel.copy()
            else:
                prev_observed_vel_state = modeled_next_vel.copy()
        else:
            prev_observed_vel_state = modeled_next_vel.copy()

        prev_modeled_vel_state = modeled_next_vel.copy()
        prev_modeled_rate_sp = corrected_rate.copy()
        prev_vel = vel.copy()

    return (
        theta_hist,
        vel_sp_theta_hist,
        acc_sp_theta_hist,
        att_sp_theta_hist,
        vel_state_theta_hist,
        prev_vel_sp_error_hist,
        prev_acc_sp_error_hist,
        prev_att_sp_error_hist,
        prev_rate_sp_error_hist,
        prev_vel_state_error_hist,
        nominal_rate_hist,
        corrected_rate_hist,
        corrected_acc_sp_hist,
        vel_dot_lpf_hist,
        vel_int_hist,
    )


def apply_rate_rls(rate_nominal, theta, prev_rate_sp_error, id_mode):
    if id_mode != "online":
        return rate_nominal.copy()

    corrected = np.zeros(3)
    for axis in range(3):
        phi = np.array([rate_nominal[axis], prev_rate_sp_error[axis], 1.0])
        y_hat = phi @ theta[axis]
        corrected[axis] = rate_nominal[axis] if not np.isfinite(y_hat) else y_hat
    return corrected


def apply_velocity_state_rls(vel, acc_sp, theta, prev_vel_state_error, id_mode):
    if id_mode != "online":
        return vel + acc_sp * dt

    vel_next = np.zeros(3)
    for axis in range(3):
        phi = np.array([vel[axis], acc_sp[axis], prev_vel_state_error[axis], 1.0])
        y_hat = phi @ theta[axis]
        vel_next[axis] = vel[axis] + acc_sp[axis] * dt if not np.isfinite(y_hat) else y_hat
    return vel_next


def main():
    global dt, horizon, prediction_start_time, prediction_interval

    parser = argparse.ArgumentParser(
        description="Prediction viewer that reproduces the identified internal model implemented in mpc_simulator.cu."
    )
    parser.add_argument("state", nargs="?", default="x")
    parser.add_argument("csv_path", nargs="?", default=default_csv)
    parser.add_argument("--profile", choices=("px4", "simulation", "current", "legacy"), default="simulation")
    parser.add_argument("--id-mode", choices=("online", "off"), default="online", help="setpoint RLS correction used by internal-model prediction")
    parser.add_argument("--horizon", type=int, default=horizon)
    parser.add_argument("--dt", type=float, default=dt)
    parser.add_argument("--prediction-start", type=float, default=prediction_start_time)
    parser.add_argument("--prediction-interval", type=float, default=prediction_interval)
    args = parser.parse_args()

    state = rate_alias.get(args.state, args.state)
    csv_path = args.csv_path
    dt = args.dt
    horizon = args.horizon
    prediction_start_time = args.prediction_start
    prediction_interval = args.prediction_interval
    apply_param_profile(args.profile)

    is_rate_plot = state in rate_names
    is_angular_state_plot = state in state_to_rate
    is_rate_compare_plot = is_rate_plot or is_angular_state_plot
    is_acc_setpoint_plot = state == "acc_setpoint"
    is_omega_setpoint_plot = state == "omega_setpoint"
    is_acc_axis_plot = state in acc_names
    is_setpoint_group_plot = is_acc_setpoint_plot or is_omega_setpoint_plot

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)

    csv_dt = dt
    if "t" not in df.columns:
        df["t"] = np.arange(len(df)) * csv_dt
    else:
        t_diff = np.diff(df["t"].to_numpy(float))
        finite_dt = t_diff[np.isfinite(t_diff) & (t_diff > 0.0)]
        if len(finite_dt) > 0:
            csv_dt = float(np.median(finite_dt))

    cur_col = f"cur_{state}"
    target_col = f"target_{state}"
    u0_col = f"u0_{state}"

    if (state not in state_index) and (not is_rate_plot) and (not is_setpoint_group_plot) and (not is_acc_axis_plot):
        print("[ERROR] state must be one of:")
        for name in state_names:
            print(" ", name)
        print("or acceleration setpoint axis:")
        for name in acc_names:
            print(" ", name)
        print("or rate setpoint name:")
        for name in rate_names:
            print(" ", name)
        print("or setpoint group:")
        for name in setpoint_group_names:
            print(" ", name)
        print("aliases: wx_sp, wy_sp, wz_sp")
        sys.exit(1)

    if is_setpoint_group_plot:
        missing_cols = []
        for col in (acc_cols if is_acc_setpoint_plot else rate_cols):
            if col not in df.columns:
                missing_cols.append(col)
        if missing_cols:
            print(f"[ERROR] missing columns: {missing_cols}")
            sys.exit(1)
    elif is_acc_axis_plot:
        acc_axis = acc_index[state]
        if acc_cols[acc_axis] not in df.columns:
            print(f"[ERROR] missing column: {acc_cols[acc_axis]}")
            sys.exit(1)
    elif is_rate_plot:
        actual_state = rate_to_state[state]
        actual_cur_col = f"cur_{actual_state}"
        if actual_cur_col not in df.columns:
            print(f"[ERROR] missing column: {actual_cur_col}")
            sys.exit(1)
    elif cur_col not in df.columns:
        print(f"[ERROR] missing column: {cur_col}")
        sys.exit(1)

    (
        rate_theta_hist,
        vel_sp_theta_hist,
        acc_sp_theta_hist,
        att_sp_theta_hist,
        _vel_state_theta_hist,
        prev_vel_sp_error_hist,
        prev_acc_sp_error_hist,
        prev_att_sp_error_hist,
        prev_rate_sp_error_hist,
        _prev_vel_state_error_hist,
        _nominal_rate_hist,
        corrected_rate_hist,
        corrected_acc_sp_hist,
        vel_dot_lpf_hist,
        vel_int_hist,
    ) = build_rls_history(df, args.id_mode)
    square_target_index_hist, square_target_change_time_hist = build_square_target_history(df)

    _, _, omega_sp_hist, internal_w_next_hist = build_simulator_history(df)

    print(f"[INFO] csv={csv_path}")
    print(f"[INFO] state={state}, profile={args.profile}, id_mode={args.id_mode}, model_dt={dt:.6f}, csv_dt={csv_dt:.6f}, horizon={horizon}")
    setpoint_model_info = "internal_px4_like_with_rls" if args.id_mode == "online" else "internal_px4_like_nominal_no_rls"
    print(f"[INFO] model=mpc_simulator_identified_internal_model, setpoints={setpoint_model_info}")
    print(
        "[INFO] square_route "
        f"waypoints={SQUARE_WAYPOINTS[:, :2].tolist()}, "
        f"hold={SQUARE_WAYPOINT_HOLD_SEC:.2f}s, threshold={SQUARE_WAYPOINT_THRESHOLD:.2f}m"
    )
    print(
        "[INFO] internal_model "
        f"vel_a={MODEL_VEL_A}, vel_b={MODEL_VEL_B}, "
        f"w_c={MODEL_W_C}, w_d={MODEL_W_D}, "
        f"pos_alpha={MODEL_POS_ALPHA}, att_beta={MODEL_ATT_BETA:.8g}"
    )
    print(
        "[INFO] params "
        f"MPC_XY_P={MPC_XY_P:.4g}, MPC_Z_P={MPC_Z_P:.4g}, "
        f"MPC_XY_VEL_P/I/D={MPC_XY_VEL_P_ACC:.4g}/{MPC_XY_VEL_I_ACC:.4g}/{MPC_XY_VEL_D_ACC:.4g}, "
        f"MPC_Z_VEL_P/I/D={MPC_Z_VEL_P_ACC:.4g}/{MPC_Z_VEL_I_ACC:.4g}/{MPC_Z_VEL_D_ACC:.4g}, "
        f"MC_R/P/Y={MC_ROLL_P:.4g}/{MC_PITCH_P:.4g}/{MC_YAW_P:.4g}"
    )

    def calc_prediction_one_row(row_idx):
        row = df.iloc[row_idx]
        s = np.zeros(len(state_names))

        for name in state_names:
            col = f"cur_{name}"
            if col in df.columns:
                s[state_index[name]] = row[col]

        q = normalize(s[0:4])
        if np.linalg.norm(q) < 1e-8:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        s[0:4] = q

        if row_idx == 0:
            prev_vel = np.array([row["cur_vx"], row["cur_vy"], row["cur_vz"]], dtype=float)
        else:
            prev_row = df.iloc[row_idx - 1]
            prev_vel = np.array([prev_row["cur_vx"], prev_row["cur_vy"], prev_row["cur_vz"]], dtype=float)

        vel_dot_lpf = vel_dot_lpf_hist[row_idx].copy()
        vel_int = vel_int_hist[row_idx].copy()
        rls_ctx = None
        if args.id_mode == "online":
            rls_ctx = {
                "rate_theta": rate_theta_hist[row_idx].copy(),
                "vel_sp_theta": vel_sp_theta_hist[row_idx].copy(),
                "acc_sp_theta": acc_sp_theta_hist[row_idx].copy(),
                "att_sp_theta": att_sp_theta_hist[row_idx].copy(),
                "prev_vel_sp_error": prev_vel_sp_error_hist[row_idx].copy(),
                "prev_acc_sp_error": prev_acc_sp_error_hist[row_idx].copy(),
                "prev_att_sp_error": prev_att_sp_error_hist[row_idx].copy(),
                "prev_rate_sp_error": prev_rate_sp_error_hist[row_idx].copy(),
            }
        route = {
            "index": int(square_target_index_hist[row_idx]),
            "row_index": int(square_target_index_hist[row_idx]),
            "change_time": float(square_target_change_time_hist[row_idx]),
        }

        pred = np.zeros((horizon + 1, len(state_names)))
        pred_omega_sp = np.full((horizon + 1, 3), np.nan, dtype=float)
        pred_acc_sp = np.full((horizon + 1, 3), np.nan, dtype=float)

        # pred[0] is the logged current state.  Then input u_h generates
        # pred[h+1], matching the controller/simulator horizon indexing.
        pred[0] = s.copy()

        for h in range(horizon):
            current_time = float(row["t"]) + h * dt
            pos_sp = get_prediction_position_setpoint(row, h, route, s, current_time)
            yaw_sp = get_prediction_yaw_setpoint(row, h, route)

            s, omega_sp, acc_sp, vel_dot_lpf, vel_int = simulator_state_step_with_setpoint_rls(
                s,
                pos_sp,
                yaw_sp,
                prev_vel,
                vel_dot_lpf,
                vel_int,
                dt,
                rls_ctx,
            )
            pred_omega_sp[h] = omega_sp.copy()
            pred_acc_sp[h] = acc_sp.copy()
            pred[h + 1] = s.copy()
            prev_vel = s[[state_index["vx"], state_index["vy"], state_index["vz"]]].copy()

        return pred, pred_omega_sp, pred_acc_sp

    segments = []
    setpoint_segments = [[], [], []]
    t_array = df["t"].to_numpy()
    start_idx = int(np.searchsorted(t_array, prediction_start_time))
    plot_every = max(1, int(round(prediction_interval / csv_dt)))

    for i in range(start_idx, len(df), plot_every):
        row = df.iloc[i]
        pred, pred_omega_sp, pred_acc_sp = calc_prediction_one_row(i)
        xs = row["t"] + np.arange(horizon + 1) * dt
        if is_acc_setpoint_plot:
            for axis in range(3):
                setpoint_segments[axis].append(np.column_stack([xs, pred_acc_sp[:, axis]]))
            continue
        elif is_acc_axis_plot:
            acc_axis = acc_index[state]
            ys = pred_acc_sp[:, acc_axis].copy()
        elif is_omega_setpoint_plot:
            for axis in range(3):
                setpoint_segments[axis].append(np.column_stack([xs, pred_omega_sp[:, axis]]))
            continue
        elif is_rate_plot:
            actual_state = rate_to_state[state]
            ys = pred[:, state_index[actual_state]].copy()
        elif is_angular_state_plot:
            ys = pred[:, state_index[state]].copy()
        else:
            ys = pred[:, state_index[state]].copy()
        segments.append(np.column_stack([xs, ys]))

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25, right=0.88)

    if is_setpoint_group_plot:
        component_colors = ["tab:blue", "tab:orange", "tab:green"]
        if is_acc_setpoint_plot:
            px4_cols = acc_cols
            calc_hist = corrected_acc_sp_hist
            component_names = ["acc_x", "acc_y", "acc_z"]
        else:
            px4_cols = rate_cols
            calc_hist = corrected_rate_hist
            component_names = ["roll_rate", "pitch_rate", "yaw_rate"]

        for axis, (name, px4_col, color) in enumerate(zip(component_names, px4_cols, component_colors)):
            ax.plot(
                df["t"],
                df[px4_col],
                color=color,
                linewidth=2.0,
                alpha=0.75,
                label=f"px4_{name}",
            )
            ax.plot(
                df["t"],
                calc_hist[:, axis],
                color=color,
                linewidth=2.0,
                linestyle="--",
                label=f"calculated_{name}",
            )
            if setpoint_segments[axis]:
                lc = LineCollection(
                    setpoint_segments[axis],
                    colors=color,
                    linewidths=1.3,
                    alpha=0.9,
                )
                ax.add_collection(lc)
                ax.plot([], [], color=color, linewidth=2, alpha=0.9, label=f"predicted_{name}")
    elif is_acc_axis_plot:
        acc_axis = acc_index[state]
        px4_col = acc_cols[acc_axis]
        ax.plot(
            df["t"],
            df[px4_col],
            color="tab:blue",
            linewidth=2.4,
            alpha=0.8,
            label=f"px4_{state}",
        )
        ax.plot(
            df["t"],
            corrected_acc_sp_hist[:, acc_axis],
            color="darkorange",
            linewidth=2.2,
            linestyle="--",
            label=f"calculated_{state}",
        )
    elif is_rate_plot:
        actual_state = rate_to_state[state]
        actual_cur_col = f"cur_{actual_state}"
        ax.plot(
            df["t"],
            df[actual_cur_col],
            color="blue",
            linewidth=3,
            label=actual_cur_col,
        )
    elif is_angular_state_plot:
        ax.plot(
            df["t"],
            df[cur_col],
            color="blue",
            linewidth=3,
            label=cur_col,
        )
    else:
        ax.plot(
            df["t"],
            df[cur_col],
            color="blue",
            linewidth=3,
            label=cur_col,
        )

        if u0_col in df.columns:
            ax.plot(
                df["t"],
                df[u0_col],
                color="gold",
                linewidth=2,
                label=u0_col,
            )

    if (not is_setpoint_group_plot) and len(segments) > 0:
        lc = LineCollection(
            segments,
            colors="red" if not is_acc_axis_plot else "tab:green",
            linewidths=1.5,
        )
        ax.add_collection(lc)
        pred_label = f"predicted_{state}"
        if is_angular_state_plot:
            pred_label = f"predicted_{state_to_rate[state]}"
        ax.plot([], [], color="red" if not is_acc_axis_plot else "tab:green", linewidth=2, label=pred_label)

    if (not is_setpoint_group_plot) and (not is_rate_plot) and (not is_angular_state_plot) and target_col in df.columns:
        ax.plot(
            df["t"],
            df[target_col],
            color="green",
            linewidth=2,
            label=target_col,
        )

    ax.set_title(state)
    ax.set_xlabel("time [s]")
    ax.grid()
    ax.legend(fontsize=14)

    ax.relim()
    ax.autoscale_view()

    t_min = 0.0
    t_max = float(df["t"].max())

    init_width = min(2.0, t_max - t_min)
    window_width = [init_width]

    slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
    time_slider = Slider(slider_ax, "time", t_min, t_max, valinit=init_width / 2.0)

    ymin, ymax = ax.get_ylim()
    y_height = [ymax - ymin]

    y_slider_ax = plt.axes([0.91, 0.2, 0.02, 0.65])
    y_slider = Slider(
        y_slider_ax,
        "y",
        ymin,
        ymax,
        valinit=0.5 * (ymin + ymax),
        orientation="vertical",
    )

    def set_xlim(center):
        half = window_width[0] / 2.0
        ax.set_xlim(center - half, center + half)
        fig.canvas.draw_idle()

    def set_ylim(center):
        half = y_height[0] / 2.0
        ax.set_ylim(center - half, center + half)
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes != ax:
            return
        if event.key == "shift":
            scale = 0.8 if event.button == "up" else 1.25
            y_height[0] *= scale
            set_ylim(y_slider.val)
        else:
            scale = 0.8 if event.button == "up" else 1.25
            window_width[0] *= scale
            set_xlim(time_slider.val)

    def on_key(event):
        if event.key == "a":
            time_slider.set_val(time_slider.val - window_width[0] * 0.1)
        elif event.key == "d":
            time_slider.set_val(time_slider.val + window_width[0] * 0.1)
        elif event.key == "w":
            window_width[0] *= 0.8
            set_xlim(time_slider.val)
        elif event.key == "p":
            window_width[0] *= 1.25
            set_xlim(time_slider.val)
        elif event.key == "j":
            y_slider.set_val(y_slider.val - y_height[0] * 0.1)
        elif event.key == "l":
            y_slider.set_val(y_slider.val + y_height[0] * 0.1)
        elif event.key == "i":
            y_height[0] *= 0.8
            set_ylim(y_slider.val)
        elif event.key == "k":
            y_height[0] *= 1.25
            set_ylim(y_slider.val)
        elif event.key == "r":
            window_width[0] = init_width
            y_height[0] = ymax - ymin
            time_slider.set_val(init_width / 2.0)
            y_slider.set_val(0.5 * (ymin + ymax))
            set_xlim(time_slider.val)
            set_ylim(y_slider.val)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    time_slider.on_changed(lambda v: set_xlim(v))
    y_slider.on_changed(lambda v: set_ylim(v))

    set_xlim(init_width / 2.0)
    set_ylim(y_slider.val)
    plt.show()


if __name__ == "__main__":
    main()
