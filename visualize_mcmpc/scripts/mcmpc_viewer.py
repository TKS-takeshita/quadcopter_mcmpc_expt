import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection


default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_20260609_003929.csv"

# ===== basic settings =====
dt = 0.02
horizon = 50
prediction_start_time = 0.1
prediction_interval = 1.0

# ===== PX4/MPC parameters =====
A_OF_GRAVITY = 9.80665
MAX_THRUST = 38.42

MPC_XY_P = 0.95
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
MC_PITCH_P = 2.00
MC_YAW_P = 2.80

MPC_VELD_LP = 5.0
ARW_GAIN = 2.0 / MPC_XY_VEL_P_ACC


state_names = [
    "e0", "e1", "e2", "e3",
    "wx", "wy", "wz",
    "x", "y", "z",
    "vx", "vy", "vz",
]
state_index = {name: i for i, name in enumerate(state_names)}

rate_names = ["roll_rate", "pitch_rate", "yaw_rate"]
rate_cols = ["px4_sp_roll_rate", "px4_sp_pitch_rate", "px4_sp_yaw_rate"]
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


def get_yaw_setpoint_from_row(row):
    if "u0_yaw" in row:
        return row["u0_yaw"]
    if {"target_e0", "target_e1", "target_e2", "target_e3"}.issubset(row.index):
        return quat_to_yaw(row["target_e0"], row["target_e1"], row["target_e2"], row["target_e3"])
    return quat_to_yaw(row["cur_e0"], row["cur_e1"], row["cur_e2"], row["cur_e3"])


def px4_nominal_setpoints(pos, vel, q, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, step_dt):
    # ==============================
    # PX4 _positionControl(): RLSなし
    # ==============================
    vel_sp_position = np.array([
        MPC_XY_P * (pos_sp[0] - pos[0]),
        MPC_XY_P * (pos_sp[1] - pos[1]),
        MPC_Z_P * (pos_sp[2] - pos[2]),
    ])

    vel_sp = vel_sp_position.copy()
    vel_sp[:2] = constrain_xy(
        vel_sp_position[:2],
        vel_sp[:2] - vel_sp_position[:2],
        MPC_XY_VEL_MAX,
    )
    vel_sp[2] = np.clip(vel_sp[2], -MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DOWN)

    # ==============================
    # PX4 _velocityControl(): RLSなし
    # ==============================
    vel_error = vel_sp - vel

    if MPC_VELD_LP > 1e-6:
        vel_dot_alpha = step_dt / (step_dt + 1.0 / (2.0 * np.pi * MPC_VELD_LP))
    else:
        vel_dot_alpha = 1.0

    vel_dot_raw = (vel - prev_vel) / step_dt
    vel_dot_lpf = vel_dot_lpf + vel_dot_alpha * (vel_dot_raw - vel_dot_lpf)
    vel_dot = vel_dot_lpf

    acc_sp = np.zeros(3)
    acc_sp[0] = MPC_XY_VEL_P_ACC * vel_error[0] + vel_int[0] - MPC_XY_VEL_D_ACC * vel_dot[0]
    acc_sp[1] = MPC_XY_VEL_P_ACC * vel_error[1] + vel_int[1] - MPC_XY_VEL_D_ACC * vel_dot[1]
    acc_sp[2] = MPC_Z_VEL_P_ACC * vel_error[2] + vel_int[2] - MPC_Z_VEL_D_ACC * vel_dot[2]

    # ==============================
    # PX4 _accelerationControl(): 姿勢setpointまでRLSなし
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

    att_sp = thrust_to_attitude(thr_sp, yaw_sp)

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
        "acc_sp": acc_sp,
        "att_sp": att_sp,
        "rate_nominal": rate_nominal,
        "thr_sp": thr_sp,
        "vel_dot_lpf": vel_dot_lpf,
        "vel_int_next": vel_int_next,
    }


def build_rate_rls_history(df, id_mode):
    n = len(df)

    theta_hist = np.zeros((n, 3, 3), dtype=float)
    prev_rate_sp_error_hist = np.zeros((n, 3), dtype=float)
    nominal_rate_hist = np.zeros((n, 3), dtype=float)
    corrected_rate_hist = np.zeros((n, 3), dtype=float)
    vel_dot_lpf_hist = np.zeros((n, 3), dtype=float)
    vel_int_hist = np.zeros((n, 3), dtype=float)

    models = [
        RLSModel(
            [1.0, 0.0, 0.0],
            lower=np.array([-20.0, -0.99, -10.0]),
            upper=np.array([20.0, 0.999, 10.0]),
        )
        for _ in range(3)
    ]

    prev_vel = df.loc[0, ["cur_vx", "cur_vy", "cur_vz"]].to_numpy(float)
    vel_dot_lpf = np.zeros(3)
    vel_int = np.zeros(3)
    prev_observed_rate_sp = np.zeros(3)
    prev_modeled_rate_sp = np.zeros(3)

    for i in range(n):
        row = df.iloc[i]

        pos = row[["cur_x", "cur_y", "cur_z"]].to_numpy(float)
        vel = row[["cur_vx", "cur_vy", "cur_vz"]].to_numpy(float)
        q = normalize(row[["cur_e0", "cur_e1", "cur_e2", "cur_e3"]].to_numpy(float))

        # ログ上のsetpoint viewerと同じく、速度・加速度・姿勢はRLSなし
        if {"u0_x", "u0_y", "u0_z"}.issubset(df.columns):
            pos_sp = row[["u0_x", "u0_y", "u0_z"]].to_numpy(float)
        elif {"target_x", "target_y", "target_z"}.issubset(df.columns):
            pos_sp = row[["target_x", "target_y", "target_z"]].to_numpy(float)
        else:
            pos_sp = pos.copy()
        yaw_sp = get_yaw_setpoint_from_row(row)

        vel_dot_lpf_hist[i] = vel_dot_lpf
        vel_int_hist[i] = vel_int
        nominal = px4_nominal_setpoints(pos, vel, q, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, dt)
        rate_nominal = nominal["rate_nominal"]
        vel_dot_lpf = nominal["vel_dot_lpf"]
        vel_int = nominal["vel_int_next"]

        prev_rate_sp_error = prev_observed_rate_sp - prev_modeled_rate_sp
        prev_rate_sp_error_hist[i] = prev_rate_sp_error
        nominal_rate_hist[i] = rate_nominal

        corrected_rate = rate_nominal.copy()
        if id_mode == "online":
            for axis in range(3):
                phi = np.array([rate_nominal[axis], prev_rate_sp_error[axis], 1.0])
                y_hat = models[axis].predict(phi)
                corrected_rate[axis] = rate_nominal[axis] if not np.isfinite(y_hat) else y_hat

        corrected_rate_hist[i] = corrected_rate
        for axis in range(3):
            theta_hist[i, axis] = models[axis].theta

        if id_mode == "online" and all(col in df.columns for col in rate_cols):
            logged_rate_sp = row[rate_cols].to_numpy(float)
            for axis in range(3):
                phi = np.array([rate_nominal[axis], prev_rate_sp_error[axis], 1.0])
                models[axis].update(phi, logged_rate_sp[axis])
            if np.all(np.isfinite(logged_rate_sp)):
                prev_observed_rate_sp = logged_rate_sp.copy()
            else:
                prev_observed_rate_sp = corrected_rate.copy()
        else:
            prev_observed_rate_sp = corrected_rate.copy()

        prev_modeled_rate_sp = corrected_rate.copy()
        prev_vel = vel.copy()

    return (
        theta_hist,
        prev_rate_sp_error_hist,
        nominal_rate_hist,
        corrected_rate_hist,
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


def main():
    global dt, horizon, prediction_start_time, prediction_interval

    parser = argparse.ArgumentParser(
        description="Prediction viewer whose angular-rate setpoint calculation matches the rate-only RLS setpoint viewer."
    )
    parser.add_argument("state", nargs="?", default="x")
    parser.add_argument("csv_path", nargs="?", default=default_csv)
    parser.add_argument("--profile", choices=("px4", "simulation", "current", "legacy"), default="simulation")
    parser.add_argument("--id-mode", choices=("online", "off"), default="online")
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

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.reset_index(drop=True)

    if "t" not in df.columns:
        df["t"] = np.arange(len(df)) * dt
    else:
        t_diff = np.diff(df["t"].to_numpy(float))
        finite_dt = t_diff[np.isfinite(t_diff) & (t_diff > 0.0)]
        if len(finite_dt) > 0:
            dt = float(np.median(finite_dt))

    cur_col = f"cur_{state}"
    target_col = f"target_{state}"
    u0_col = f"u0_{state}"

    if (state not in state_index) and (not is_rate_plot):
        print("[ERROR] state must be one of:")
        for name in state_names:
            print(" ", name)
        print("or rate setpoint name:")
        for name in rate_names:
            print(" ", name)
        print("aliases: wx_sp, wy_sp, wz_sp")
        sys.exit(1)

    if is_rate_plot:
        actual_state = rate_to_state[state]
        actual_cur_col = f"cur_{actual_state}"
        if actual_cur_col not in df.columns:
            print(f"[ERROR] missing column: {actual_cur_col}")
            sys.exit(1)
    elif cur_col not in df.columns:
        print(f"[ERROR] missing column: {cur_col}")
        sys.exit(1)

    (
        theta_hist,
        prev_rate_sp_error_hist,
        nominal_rate_hist,
        corrected_rate_hist,
        vel_dot_lpf_hist,
        vel_int_hist,
    ) = build_rate_rls_history(df, args.id_mode)

    print(f"[INFO] csv={csv_path}")
    print(f"[INFO] state={state}, profile={args.profile}, id_mode={args.id_mode}, dt={dt:.6f}, horizon={horizon}")
    print("[INFO] rate_setpoint_model=formula_plus_rate_rls")
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

        # 予測の初期状態は必ず現在状態 cur_* から開始
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
        theta = theta_hist[row_idx].copy()
        prev_rate_sp_error = prev_rate_sp_error_hist[row_idx].copy()

        pred = np.zeros((horizon + 1, len(state_names)))
        pred_rate = np.zeros((horizon + 1, 3), dtype=float)
        pred[0] = s.copy()
        pred_rate[0] = corrected_rate_hist[row_idx].copy()

        for h in range(horizon):
            x_ref = row[f"u{h}_x"] if f"u{h}_x" in df.columns else row.get("u0_x", row.get("target_x", s[state_index["x"]]))
            y_ref = row[f"u{h}_y"] if f"u{h}_y" in df.columns else row.get("u0_y", row.get("target_y", s[state_index["y"]]))
            z_ref = row[f"u{h}_z"] if f"u{h}_z" in df.columns else row.get("u0_z", row.get("target_z", s[state_index["z"]]))
            pos_sp = np.array([x_ref, y_ref, z_ref], dtype=float)
            yaw_sp = row[f"u{h}_yaw"] if f"u{h}_yaw" in df.columns else get_yaw_setpoint_from_row(row)

            pos = s[[state_index["x"], state_index["y"], state_index["z"]]]
            vel = s[[state_index["vx"], state_index["vy"], state_index["vz"]]]
            q = normalize(s[0:4])

            nominal = px4_nominal_setpoints(pos, vel, q, pos_sp, yaw_sp, prev_vel, vel_dot_lpf, vel_int, dt)
            rate_nominal = nominal["rate_nominal"]
            rate_sp = apply_rate_rls(rate_nominal, theta, prev_rate_sp_error, args.id_mode)
            pred_rate[h + 1] = rate_sp.copy()

            # 未来予測中はPX4ログの未来値がないため、予測値を仮想PX4出力として扱う。
            prev_rate_sp_error = np.zeros(3)

            acc_sp = nominal["acc_sp"]
            vel_dot_lpf = nominal["vel_dot_lpf"]
            vel_int = nominal["vel_int_next"]

            step = dt / 2.0
            for _ in range(2):
                s_old = s.copy()
                q_old = normalize(s_old[0:4])

                s[state_index["x"]] += s_old[state_index["vx"]] * step
                s[state_index["y"]] += s_old[state_index["vy"]] * step
                s[state_index["z"]] += s_old[state_index["vz"]] * step

                s[state_index["vx"]] += acc_sp[0] * step
                s[state_index["vy"]] += acc_sp[1] * step
                s[state_index["vz"]] += acc_sp[2] * step

                q_new = integrate_quat(q_old, rate_sp, step)
                s[0:4] = q_new
                s[state_index["wx"]] = rate_sp[0]
                s[state_index["wy"]] = rate_sp[1]
                s[state_index["wz"]] = rate_sp[2]

            pred[h + 1] = s.copy()
            prev_vel = s[[state_index["vx"], state_index["vy"], state_index["vz"]]].copy()

        return pred, pred_rate

    segments = []
    t_array = df["t"].to_numpy()
    start_idx = int(np.searchsorted(t_array, prediction_start_time))
    plot_every = max(1, int(round(prediction_interval / dt)))

    for i in range(start_idx, len(df), plot_every):
        row = df.iloc[i]
        pred, pred_rate = calc_prediction_one_row(i)
        xs = row["t"] + np.arange(horizon + 1) * dt
        if is_rate_plot:
            ys = pred_rate[:, rate_index[state]].copy()
        elif is_angular_state_plot:
            rate_name_for_state = state_to_rate[state]
            ys = pred_rate[:, rate_index[rate_name_for_state]].copy()
        else:
            ys = pred[:, state_index[state]].copy()
        segments.append(np.column_stack([xs, ys]))

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25, right=0.88)

    if is_rate_plot:
        actual_state = rate_to_state[state]
        actual_cur_col = f"cur_{actual_state}"
        rate_axis = rate_index[state]
        ax.plot(
            df["t"],
            df[actual_cur_col],
            color="blue",
            linewidth=3,
            label=actual_cur_col,
        )
        ax.plot(
            df["t"],
            corrected_rate_hist[:, rate_axis],
            color="darkorange",
            linewidth=2,
            alpha=0.9,
            label=f"calculated_{state}",
        )
        px4_rate_col = rate_cols[rate_axis]
        if px4_rate_col in df.columns:
            ax.plot(
                df["t"],
                df[px4_rate_col],
                color="purple",
                linewidth=2,
                alpha=0.8,
                label=px4_rate_col,
            )
    elif is_angular_state_plot:
        ax.plot(
            df["t"],
            df[cur_col],
            color="blue",
            linewidth=3,
            label=cur_col,
        )
        rate_name_for_state = state_to_rate[state]
        rate_axis = rate_index[rate_name_for_state]
        ax.plot(
            df["t"],
            corrected_rate_hist[:, rate_axis],
            color="darkorange",
            linewidth=2,
            alpha=0.9,
            label=f"calculated_{rate_name_for_state}",
        )
        px4_rate_col = rate_cols[rate_axis]
        if px4_rate_col in df.columns:
            ax.plot(
                df["t"],
                df[px4_rate_col],
                color="purple",
                linewidth=2,
                alpha=0.8,
                label=px4_rate_col,
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

    if len(segments) > 0:
        lc = LineCollection(
            segments,
            colors="red",
            linewidths=1.5,
        )
        ax.add_collection(lc)
        pred_label = f"predicted_{state}"
        if is_angular_state_plot:
            pred_label = f"predicted_{state_to_rate[state]}"
        ax.plot([], [], color="red", linewidth=2, label=pred_label)

    if (not is_rate_plot) and (not is_angular_state_plot) and target_col in df.columns:
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
