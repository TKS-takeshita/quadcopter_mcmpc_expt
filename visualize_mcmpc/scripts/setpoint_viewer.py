import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_20260609_003929.csv"

dt = 0.02
position_source = "auto"
velocity_source = "auto"
position_source_by_row = None
velocity_sp_model = None
acc_sp_model = None
att_sp_model = None
rate_sp_model = None
identification_mode = "online"

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

MC_YAW_WEIGHT = 0.4
MC_ROLL_P = 4.00
MC_PITCH_P = 2.00
MC_YAW_P = 2.8

MPC_VEL_LP = 0.0
MPC_VELD_LP = 5.0

MC_ROLLRATE_P = 0.15
MC_PITCHRATE_P = 0.15
MC_YAWRATE_P = 0.20
MC_ROLLRATE_D = 0.0030
MC_PITCHRATE_D = 0.0030
MC_YAWRATE_D = 0.00
MC_ROLLRATE_I = 0.00
MC_PITCHRATE_I = 0.00
MC_YAWRATE_I = 0.00

ARW_GAIN = 2.0 / MPC_XY_VEL_P_ACC


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


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def wrap_pi(x):
    return np.arctan2(np.sin(x), np.cos(x))


def quat_to_yaw(q0, q1, q2, q3):
    return np.arctan2(
        2.0 * (q0 * q3 + q1 * q2),
        q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    )


def apply_param_profile(profile):
    global MPC_XY_P, MPC_Z_P
    global MPC_XY_VEL_P_ACC, MPC_XY_VEL_I_ACC, MPC_XY_VEL_D_ACC
    global MPC_Z_VEL_P_ACC, MPC_Z_VEL_I_ACC, MPC_Z_VEL_D_ACC
    global MC_ROLL_P, MC_PITCH_P, MC_YAW_P

    if profile == "px4":
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
    elif profile == "simulation":
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


def choose_position_source(df, requested_source):
    if requested_source != "auto":
        return requested_source

    if {"target_x", "target_y", "target_z"}.issubset(df.columns):
        return "target"

    if {"u0_x", "u0_y", "u0_z"}.issubset(df.columns):
        return "u0"

    return "u0"


def choose_position_source_by_log_fit(df, requested_source):
    if requested_source != "auto":
        return requested_source

    needed = {"px4_sp_vx", "px4_sp_vy", "px4_sp_vz", "cur_x", "cur_y", "cur_z"}
    if not needed.issubset(df.columns):
        return "u0"

    candidates = []
    for source in ("u0", "target"):
        if source == "u0":
            cols = ("u0_x", "u0_y", "u0_z")
        else:
            cols = ("target_x", "target_y", "target_z")

        if not set(cols).issubset(df.columns):
            continue

        pos_sp = df.loc[:, cols].to_numpy(float)
        pos = df.loc[:, ("cur_x", "cur_y", "cur_z")].to_numpy(float)
        vel_px4 = df.loc[:, ("px4_sp_vx", "px4_sp_vy", "px4_sp_vz")].to_numpy(float)
        vel_calc = (pos_sp - pos) * np.array([MPC_XY_P, MPC_XY_P, MPC_Z_P])
        vel_calc[:, 0:2] = np.clip(vel_calc[:, 0:2], -MPC_XY_VEL_MAX, MPC_XY_VEL_MAX)
        vel_calc[:, 2] = np.clip(vel_calc[:, 2], -MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DOWN)

        mask = np.isfinite(vel_px4) & np.isfinite(vel_calc)
        if np.count_nonzero(mask) < 20:
            continue
        rmse = np.sqrt(np.nanmean((vel_calc[mask] - vel_px4[mask]) ** 2))
        candidates.append((rmse, source))

    if not candidates:
        return "u0"

    return min(candidates)[1]


def choose_position_source_by_row(df, requested_source):
    if requested_source != "auto":
        return np.full(len(df), requested_source, dtype=object)

    required = {
        "cur_x", "cur_y", "cur_z",
        "u0_x", "u0_y", "u0_z",
        "target_x", "target_y", "target_z",
        "px4_sp_vx", "px4_sp_vy", "px4_sp_vz",
    }
    if not required.issubset(df.columns):
        return np.full(len(df), choose_position_source_by_log_fit(df, requested_source), dtype=object)

    pos = df.loc[:, ("cur_x", "cur_y", "cur_z")].to_numpy(float)
    vel_px4 = df.loc[:, ("px4_sp_vx", "px4_sp_vy", "px4_sp_vz")].to_numpy(float)
    gains = np.array([MPC_XY_P, MPC_XY_P, MPC_Z_P])

    scores = {}
    for source, cols in {
        "u0": ("u0_x", "u0_y", "u0_z"),
        "target": ("target_x", "target_y", "target_z"),
    }.items():
        sp = df.loc[:, cols].to_numpy(float)
        vel = (sp - pos) * gains
        vel[:, 0:2] = np.clip(vel[:, 0:2], -MPC_XY_VEL_MAX, MPC_XY_VEL_MAX)
        vel[:, 2] = np.clip(vel[:, 2], -MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DOWN)
        valid = np.isfinite(vel_px4) & np.isfinite(vel)
        err = np.where(valid, vel - vel_px4, np.nan)
        valid_count = np.sum(valid, axis=1)
        err_sum = np.nansum(err * err, axis=1)
        scores[source] = np.full(len(df), np.nan)
        np.divide(err_sum, valid_count, out=scores[source], where=valid_count > 0)

    selected = np.where(scores["target"] <= scores["u0"], "target", "u0").astype(object)
    finite_score = np.isfinite(scores["target"]) | np.isfinite(scores["u0"])
    fallback = choose_position_source_by_log_fit(df, requested_source)

    last = fallback
    for i in range(len(selected)):
        if finite_score[i]:
            last = selected[i]
        else:
            selected[i] = last

    return selected


def position_setpoint_array(df):
    if position_source == "row" and position_source_by_row is not None:
        u0 = df.loc[:, ("u0_x", "u0_y", "u0_z")].to_numpy(float)
        target = df.loc[:, ("target_x", "target_y", "target_z")].to_numpy(float)
        use_target = np.asarray(position_source_by_row) == "target"
        return np.where(use_target[:, None], target, u0)

    if position_source == "target":
        return df.loc[:, ("target_x", "target_y", "target_z")].to_numpy(float)

    return df.loc[:, ("u0_x", "u0_y", "u0_z")].to_numpy(float)


def get_position_setpoint(row, source):
    if source == "row":
        source = row.get("_viewer_position_source", "u0")

    if source == "target":
        return np.array([
            row["target_x"],
            row["target_y"],
            row["target_z"],
        ])

    return np.array([
        row["u0_x"],
        row["u0_y"],
        row["u0_z"],
    ])


def get_yaw_setpoint(row, source):
    if source == "row":
        source = row.get("_viewer_position_source", "u0")

    if source == "target":
        return quat_to_yaw(
            row.get("target_e0", 1.0),
            row.get("target_e1", 0.0),
            row.get("target_e2", 0.0),
            row.get("target_e3", 0.0),
        )

    return row["u0_yaw"]


def safe_lstsq(x, y, default, min_samples=20, lower=None, upper=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if np.count_nonzero(mask) < min_samples:
        return default

    coeff, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
    coeff = np.asarray(coeff, dtype=float)
    if lower is not None or upper is not None:
        lo = -np.inf if lower is None else lower
        hi = np.inf if upper is None else upper
        coeff = np.clip(coeff, lo, hi)
    return coeff


def velocity_lpf_derivative(df):
    vel = df.loc[:, ("cur_vx", "cur_vy", "cur_vz")].to_numpy(float)
    vel_dot_lpf = np.zeros_like(vel)
    if MPC_VELD_LP > 1e-6:
        alpha = dt / (dt + 1.0 / (2.0 * np.pi * MPC_VELD_LP))
    else:
        alpha = 1.0

    state = np.zeros(3)
    prev_vel = vel[0].copy()
    for i in range(len(df)):
        if i == 0:
            vel_dot_lpf[i] = 0.0
        else:
            raw = (vel[i] - prev_vel) / dt
            state = state + alpha * (raw - state)
            vel_dot_lpf[i] = state
        prev_vel = vel[i].copy()
    return vel_dot_lpf


def fit_position_gains(df):
    global MPC_XY_P, MPC_Z_P

    pos = df.loc[:, ("cur_x", "cur_y", "cur_z")].to_numpy(float)
    pos_sp = position_setpoint_array(df)
    err = pos_sp - pos
    vel_px4 = df.loc[:, ("px4_sp_vx", "px4_sp_vy", "px4_sp_vz")].to_numpy(float)

    xy_x = np.concatenate([err[:, 0], err[:, 1]])
    xy_y = np.concatenate([vel_px4[:, 0], vel_px4[:, 1]])
    k_xy = safe_lstsq(xy_x, xy_y, np.array([MPC_XY_P]), lower=0.0, upper=5.0)[0]
    k_z = safe_lstsq(err[:, 2], vel_px4[:, 2], np.array([MPC_Z_P]), lower=-5.0, upper=5.0)[0]

    MPC_XY_P = float(k_xy)
    MPC_Z_P = float(k_z)


def fit_velocity_setpoint_model(df):
    global velocity_sp_model

    pos = df.loc[:, ("cur_x", "cur_y", "cur_z")].to_numpy(float)
    pos_sp = position_setpoint_array(df)
    err = pos_sp - pos
    vel_px4 = df.loc[:, ("px4_sp_vx", "px4_sp_vy", "px4_sp_vz")].to_numpy(float)

    # Model:
    #   vel_sp[k] = k_err * pos_err[k]
    #             + k_prev_err * (px4_vel_sp[k-1] - model_vel_sp[k-1])
    #             + bias
    # Here, model_vel_sp[k-1] is the nominal position-P velocity setpoint
    # before RLS correction. This makes the second term a correction based
    # on the previous mismatch between PX4 and this model.
    base_vel_sp = err * np.array([MPC_XY_P, MPC_XY_P, MPC_Z_P])
    base_vel_sp[:, 0:2] = np.clip(base_vel_sp[:, 0:2], -MPC_XY_VEL_MAX, MPC_XY_VEL_MAX)
    base_vel_sp[:, 2] = np.clip(base_vel_sp[:, 2], -MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DOWN)

    prev_sp_error = vel_px4[:-1] - base_vel_sp[:-1]

    coeffs = np.zeros((3, 3), dtype=float)
    for axis in range(3):
        y = vel_px4[1:, axis]
        x = np.vstack([err[1:, axis], prev_sp_error[:, axis], np.ones(len(df) - 1)]).T
        default = np.array([
            MPC_XY_P if axis < 2 else MPC_Z_P,
            0.0,
            0.0,
        ])
        coeffs[axis] = safe_lstsq(
            x,
            y,
            default,
            lower=np.array([-10.0, -10.0, -10.0]),
            upper=np.array([10.0, 10.0, 10.0]),
        )

    velocity_sp_model = coeffs


def fit_acc_setpoint_model(df):
    global acc_sp_model

    vel = df.loc[:, ("cur_vx", "cur_vy", "cur_vz")].to_numpy(float)
    vel_sp = df.loc[:, ("px4_sp_vx", "px4_sp_vy", "px4_sp_vz")].to_numpy(float)
    acc_px4 = df.loc[:, ("px4_sp_acc_x", "px4_sp_acc_y", "px4_sp_acc_z")].to_numpy(float)
    vel_dot = velocity_lpf_derivative(df)
    vel_error = vel_sp - vel

    # PX4 velocity integral basis reconstructed from the logged velocity setpoint.
    integ_basis = np.zeros_like(vel_error)
    running = np.zeros(3)
    for i in range(len(df)):
        integ_basis[i] = running
        e = np.nan_to_num(vel_error[i], nan=0.0, posinf=0.0, neginf=0.0)
        running[0] += e[0] * MPC_XY_VEL_I_ACC * dt
        running[1] += e[1] * MPC_XY_VEL_I_ACC * dt
        running[2] += e[2] * MPC_Z_VEL_I_ACC * dt
        running[2] = np.clip(running[2], -A_OF_GRAVITY, A_OF_GRAVITY)

    # Nominal PX4-like acceleration before RLS correction.
    base_acc_sp = np.zeros_like(vel_error)
    base_acc_sp[:, 0] = (
        MPC_XY_VEL_P_ACC * vel_error[:, 0]
        + integ_basis[:, 0]
        - MPC_XY_VEL_D_ACC * vel_dot[:, 0]
    )
    base_acc_sp[:, 1] = (
        MPC_XY_VEL_P_ACC * vel_error[:, 1]
        + integ_basis[:, 1]
        - MPC_XY_VEL_D_ACC * vel_dot[:, 1]
    )
    base_acc_sp[:, 2] = (
        MPC_Z_VEL_P_ACC * vel_error[:, 2]
        + integ_basis[:, 2]
        - MPC_Z_VEL_D_ACC * vel_dot[:, 2]
    )

    # Model:
    #   acc_sp[k] = k_p * vel_error[k]
    #             + k_i * vel_int[k]
    #             + k_d * (-vel_dot[k])
    #             + k_prev_err * (px4_acc_sp[k-1] - model_acc_sp[k-1])
    #             + bias
    prev_acc_error = acc_px4[:-1] - base_acc_sp[:-1]

    coeffs = np.zeros((3, 5), dtype=float)
    defaults = [
        np.array([MPC_XY_VEL_P_ACC, 1.0, MPC_XY_VEL_D_ACC, 0.0, 0.0]),
        np.array([MPC_XY_VEL_P_ACC, 1.0, MPC_XY_VEL_D_ACC, 0.0, 0.0]),
        np.array([MPC_Z_VEL_P_ACC, 1.0, MPC_Z_VEL_D_ACC, 0.0, 0.0]),
    ]

    for axis in range(3):
        y = acc_px4[1:, axis]
        x = np.vstack([
            vel_error[1:, axis],
            integ_basis[1:, axis],
            -vel_dot[1:, axis],
            prev_acc_error[:, axis],
            np.ones(len(df) - 1),
        ]).T
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1) & (np.abs(y) < 50.0)
        coeffs[axis] = safe_lstsq(
            x[mask],
            y[mask],
            defaults[axis],
            lower=np.array([-30.0, -10.0, -10.0, -10.0, -30.0]),
            upper=np.array([30.0, 10.0, 10.0, 10.0, 30.0]),
        )

    acc_sp_model = coeffs


def fit_rate_gains(df):
    global MC_ROLL_P, MC_PITCH_P, MC_YAW_P

    saved = (MC_ROLL_P, MC_PITCH_P, MC_YAW_P)
    MC_ROLL_P = 1.0
    MC_PITCH_P = 1.0
    MC_YAW_P = 1.0
    unit_calc = calc_px4_like_setpoints(df)
    MC_ROLL_P, MC_PITCH_P, MC_YAW_P = saved

    fits = []
    for key, col, default in [
        ("roll_rate", "px4_sp_roll_rate", saved[0]),
        ("pitch_rate", "px4_sp_pitch_rate", saved[1]),
        ("yaw_rate", "px4_sp_yaw_rate", saved[2]),
    ]:
        base = unit_calc[key]
        actual = df[col].to_numpy(float)
        gain = safe_lstsq(base, actual, np.array([default]), min_samples=20, lower=-20.0, upper=20.0)[0]
        fits.append(float(gain))

    MC_ROLL_P, MC_PITCH_P, MC_YAW_P = fits


def fit_att_setpoint_model(df):
    global att_sp_model

    saved = att_sp_model
    att_sp_model = None
    raw_calc = calc_px4_like_setpoints(df)
    att_sp_model = saved

    raw_q = np.vstack([
        raw_calc["q0"],
        raw_calc["q1"],
        raw_calc["q2"],
        raw_calc["q3"],
    ]).T

    px4_q = df.loc[:, ("px4_sp_q0", "px4_sp_q1", "px4_sp_q2", "px4_sp_q3")].to_numpy(float)
    aligned_px4_q = px4_q.copy()
    for i in range(len(df)):
        aligned_px4_q[i] = align_quat_sign(px4_q[i], raw_q[i])

    # Model:
    #   q_sp_component[k] = k_raw * q_nominal_component[k]
    #                     + k_prev_err * (px4_q_component[k-1] - model_q_component[k-1])
    #                     + bias
    # The quaternion is normalized after applying this component-wise correction.
    prev_q_error = aligned_px4_q[:-1] - raw_q[:-1]

    coeffs = np.zeros((4, 3), dtype=float)
    for axis in range(4):
        y = aligned_px4_q[1:, axis]
        x = np.vstack([raw_q[1:, axis], prev_q_error[:, axis], np.ones(len(df) - 1)]).T
        coeffs[axis] = safe_lstsq(
            x,
            y,
            np.array([1.0, 0.0, 0.0]),
            lower=np.array([-5.0, -10.0, -5.0]),
            upper=np.array([5.0, 10.0, 5.0]),
        )

    att_sp_model = coeffs


def fit_rate_setpoint_model(df):
    global rate_sp_model

    saved = rate_sp_model
    rate_sp_model = None
    raw_calc = calc_px4_like_setpoints(df)
    rate_sp_model = saved

    # Model:
    #   rate_sp[k] = k_raw * rate_nominal[k]
    #              + k_prev_err * (px4_rate_sp[k-1] - model_rate_sp[k-1])
    #              + bias
    # The second term uses the previous mismatch between PX4 and this model,
    # not the previous PX4 rate setpoint itself.
    coeffs = np.zeros((3, 3), dtype=float)
    for axis, key, col in [
        (0, "roll_rate", "px4_sp_roll_rate"),
        (1, "pitch_rate", "px4_sp_pitch_rate"),
        (2, "yaw_rate", "px4_sp_yaw_rate"),
    ]:
        actual = df[col].to_numpy(float)
        raw = raw_calc[key]
        prev_rate_error = actual[:-1] - raw[:-1]
        y = actual[1:]
        x = np.vstack([raw[1:], prev_rate_error, np.ones(len(df) - 1)]).T
        coeffs[axis] = safe_lstsq(
            x,
            y,
            np.array([1.0, 0.0, 0.0]),
            lower=np.array([-20.0, -10.0, -10.0]),
            upper=np.array([20.0, 10.0, 10.0]),
        )

    rate_sp_model = coeffs


def fit_params_to_log(df):
    fit_position_gains(df)
    fit_velocity_setpoint_model(df)
    fit_acc_setpoint_model(df)
    fit_att_setpoint_model(df)
    fit_rate_gains(df)
    fit_rate_setpoint_model(df)


def estimate_plot_delay(t, calc_y, px4_y, search_sec=0.5):
    step = max(dt, 1e-3)
    best_rmse = np.inf
    best_delay = 0.0
    for delay in np.arange(-search_sec, search_sec + 0.5 * step, step):
        shifted = np.interp(t, t + delay, calc_y, left=np.nan, right=np.nan)
        mask = np.isfinite(shifted) & np.isfinite(px4_y)
        if np.count_nonzero(mask) < 50:
            continue
        rmse = np.sqrt(np.mean((shifted[mask] - px4_y[mask]) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_delay = float(delay)
    return best_delay


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

    y_c = np.array([
        -np.sin(yaw_sp),
        np.cos(yaw_sp),
        0.0
    ])

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


def calc_px4_like_setpoints(df):
    n = len(df)

    calc = {
        "vx": np.full(n, np.nan),
        "vy": np.full(n, np.nan),
        "vz": np.full(n, np.nan),

        "acc_x": np.full(n, np.nan),
        "acc_y": np.full(n, np.nan),
        "acc_z": np.full(n, np.nan),

        "q0": np.full(n, np.nan),
        "q1": np.full(n, np.nan),
        "q2": np.full(n, np.nan),
        "q3": np.full(n, np.nan),

        "roll_rate": np.full(n, np.nan),
        "pitch_rate": np.full(n, np.nan),
        "yaw_rate": np.full(n, np.nan),

        "thrust_x": np.full(n, np.nan),
        "thrust_y": np.full(n, np.nan),
        "thrust_z": np.full(n, np.nan),
    }

    vel_int = np.zeros(3)

    prev_vel = np.array([
        df.loc[0, "cur_vx"],
        df.loc[0, "cur_vy"],
        df.loc[0, "cur_vz"],
    ])

    # ===== velocity derivative LPF state =====
    vel_dot_lpf = np.zeros(3)

    if MPC_VELD_LP > 1e-6:
        vel_dot_alpha = dt / (dt + 1.0 / (2.0 * np.pi * MPC_VELD_LP))
    else:
        vel_dot_alpha = 1.0

    prev_modeled_vel_sp = np.zeros(3)
    prev_observed_vel_sp = np.zeros(3)
    prev_velocity_sp_error = np.zeros(3)
    prev_modeled_rate_sp = np.zeros(3)
    prev_observed_rate_sp = np.zeros(3)
    prev_rate_sp_error = np.zeros(3)
    online_velocity_models = None
    online_acc_models = None
    online_att_models = None
    online_rate_models = None
    current_velocity_sp_phi = None
    current_acc_sp_phi = None
    current_att_sp_phi = None
    current_rate_sp_phi = None
    prev_modeled_acc_sp = np.zeros(3)
    prev_observed_acc_sp = np.zeros(3)
    prev_modeled_att_sp = np.array([1.0, 0.0, 0.0, 0.0])
    prev_observed_att_sp = np.array([1.0, 0.0, 0.0, 0.0])

    if identification_mode == "online":
        # Only angular-rate setpoint is identified by RLS.
        # Velocity, acceleration, and attitude setpoints remain PX4-like nominal calculations
        # so they can be compared against the logged PX4 setpoints without RLS correction.
        online_rate_models = [
            RLSModel(
                [1.0, 0.0, 0.0],
                lower=np.array([-20.0, -10.0, -10.0]),
                upper=np.array([20.0, 10.0, 10.0]),
            )
            for _ in range(3)
        ]

    for i in range(n):
        row = df.iloc[i]

        q = np.array([
            row["cur_e0"],
            row["cur_e1"],
            row["cur_e2"],
            row["cur_e3"],
        ])

        pos = np.array([
            row["cur_x"],
            row["cur_y"],
            row["cur_z"],
        ])

        vel = np.array([
            row["cur_vx"],
            row["cur_vy"],
            row["cur_vz"],
        ])

        pos_sp = get_position_setpoint(row, position_source)
        yaw_sp = get_yaw_setpoint(row, position_source)

        # ==============================
        # PX4 _positionControl()
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
            MPC_XY_VEL_MAX
        )

        vel_sp[2] = np.clip(
            vel_sp[2],
            -MPC_Z_VEL_MAX_UP,
            MPC_Z_VEL_MAX_DOWN
        )

        if online_velocity_models is not None:
            pos_err = pos_sp - pos
            modeled_vel_sp = np.zeros(3)
            prev_velocity_sp_error = prev_observed_vel_sp - prev_modeled_vel_sp
            current_velocity_sp_phi = np.zeros((3, 3), dtype=float)
            for axis in range(3):
                phi = np.array([pos_err[axis], prev_velocity_sp_error[axis], 1.0])
                current_velocity_sp_phi[axis] = phi
                y_hat = online_velocity_models[axis].predict(phi)
                modeled_vel_sp[axis] = vel_sp[axis] if not np.isfinite(y_hat) else y_hat
            vel_sp = modeled_vel_sp
            prev_modeled_vel_sp = modeled_vel_sp.copy()
        elif velocity_sp_model is not None:
            pos_err = pos_sp - pos
            modeled_vel_sp = np.zeros(3)
            prev_for_model = prev_modeled_vel_sp
            prev_logged_vel_sp = prev_observed_vel_sp
            if i > 0:
                logged_prev = np.array([
                    df.iloc[i - 1].get("px4_sp_vx", np.nan),
                    df.iloc[i - 1].get("px4_sp_vy", np.nan),
                    df.iloc[i - 1].get("px4_sp_vz", np.nan),
                ], dtype=float)
                if np.all(np.isfinite(logged_prev)):
                    prev_logged_vel_sp = logged_prev

            prev_velocity_sp_error = prev_logged_vel_sp - prev_for_model
            for axis in range(3):
                k_err, k_prev_err, bias = velocity_sp_model[axis]
                modeled_vel_sp[axis] = (
                    k_err * pos_err[axis]
                    + k_prev_err * prev_velocity_sp_error[axis]
                    + bias
                )
            vel_sp = modeled_vel_sp
            prev_modeled_vel_sp = modeled_vel_sp.copy()
            prev_observed_vel_sp = prev_logged_vel_sp.copy()

        vel_ctrl_sp = vel_sp.copy()
        if velocity_source == "logged":
            logged_vel_sp = np.array([
                row.get("px4_sp_vx", np.nan),
                row.get("px4_sp_vy", np.nan),
                row.get("px4_sp_vz", np.nan),
            ], dtype=float)
            if np.all(np.isfinite(logged_vel_sp)):
                vel_ctrl_sp = logged_vel_sp

        # ==============================
        # PX4 _velocityControl()
        # acc_sp = P * vel_error + vel_int - D * vel_dot
        # ==============================
        vel_error = vel_ctrl_sp - vel

        # ===== modified: raw derivative -> LPF derivative =====
        if i == 0:
            vel_dot = np.zeros(3)
            vel_dot_lpf[:] = 0.0
        else:
            vel_dot_raw = (vel - prev_vel) / dt
            vel_dot_lpf = vel_dot_lpf + vel_dot_alpha * (vel_dot_raw - vel_dot_lpf)
            vel_dot = vel_dot_lpf

        acc_sp_nominal = np.zeros(3)

        acc_sp_nominal[0] = (
            MPC_XY_VEL_P_ACC * vel_error[0]
            + vel_int[0]
            - MPC_XY_VEL_D_ACC * vel_dot[0]
        )

        acc_sp_nominal[1] = (
            MPC_XY_VEL_P_ACC * vel_error[1]
            + vel_int[1]
            - MPC_XY_VEL_D_ACC * vel_dot[1]
        )

        acc_sp_nominal[2] = (
            MPC_Z_VEL_P_ACC * vel_error[2]
            + vel_int[2]
            - MPC_Z_VEL_D_ACC * vel_dot[2]
        )

        acc_sp = acc_sp_nominal.copy()
        prev_acc_sp_error = prev_observed_acc_sp - prev_modeled_acc_sp

        if online_acc_models is not None:
            current_acc_sp_phi = np.zeros((3, 5), dtype=float)
            modeled_acc_sp = np.zeros(3)
            for axis in range(3):
                phi = np.array([
                    vel_error[axis],
                    vel_int[axis],
                    -vel_dot[axis],
                    prev_acc_sp_error[axis],
                    1.0,
                ])
                current_acc_sp_phi[axis] = phi
                y_hat = online_acc_models[axis].predict(phi)
                modeled_acc_sp[axis] = acc_sp_nominal[axis] if not np.isfinite(y_hat) else y_hat
            acc_sp = modeled_acc_sp
            prev_modeled_acc_sp = modeled_acc_sp.copy()
        elif acc_sp_model is not None:
            modeled_acc_sp = np.zeros(3)
            for axis in range(3):
                k_p, k_i, k_d, k_prev_err, bias = acc_sp_model[axis]
                modeled_acc_sp[axis] = (
                    k_p * vel_error[axis]
                    + k_i * vel_int[axis]
                    + k_d * (-vel_dot[axis])
                    + k_prev_err * prev_acc_sp_error[axis]
                    + bias
                )
            acc_sp = modeled_acc_sp
            prev_modeled_acc_sp = modeled_acc_sp.copy()

        # ==============================
        # PX4 _accelerationControl()
        # decouple horizontal and vertical acceleration = true
        # ==============================
        z_specific_force = -A_OF_GRAVITY

        body_z = np.array([
            -acc_sp[0],
            -acc_sp[1],
            -z_specific_force,
        ])
        body_z = normalize(body_z)
        body_z = limit_tilt(body_z, MPC_TILT_MAX)

        thrust_ned_z = acc_sp[2] * (MPC_THR_HOVER / A_OF_GRAVITY) - MPC_THR_HOVER
        cos_ned_body = np.dot(np.array([0.0, 0.0, 1.0]), body_z)

        if abs(cos_ned_body) < 1e-6:
            cos_ned_body = 1e-6

        collective_thrust = min(thrust_ned_z / cos_ned_body, -MPC_THR_MIN)

        thr_sp = body_z * collective_thrust

        # ==============================
        # PX4 thrust saturation
        # ==============================
        thrust_sp_xy_norm = np.linalg.norm(thr_sp[:2])
        thrust_max_squared = MPC_THR_MAX * MPC_THR_MAX

        allocated_horizontal_thrust = min(thrust_sp_xy_norm, MPC_THR_XY_MARGIN)
        thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust ** 2

        thr_sp[2] = max(thr_sp[2], -np.sqrt(max(0.0, thrust_z_max_squared)))

        thrust_max_xy_squared = thrust_max_squared - thr_sp[2] ** 2
        thrust_max_xy = np.sqrt(max(0.0, thrust_max_xy_squared))

        if thrust_sp_xy_norm > thrust_max_xy and thrust_sp_xy_norm > 1e-8:
            thr_sp[:2] = thr_sp[:2] / thrust_sp_xy_norm * thrust_max_xy

        # ==============================
        # PX4 anti-windup
        # Z方向のみ残す
        # XY ARWは produced acceleration の再現が不確実なため無効化
        # ==============================
        if (thr_sp[2] >= -MPC_THR_MIN and vel_error[2] >= 0.0) or \
           (thr_sp[2] <= -MPC_THR_MAX and vel_error[2] <= 0.0):
            vel_error[2] = 0.0

        acc_sp_xy_produced = thr_sp[:2] * (A_OF_GRAVITY / MPC_THR_HOVER)

        # XY ARW disabled
        # if thrust_sp_xy_norm > thrust_max_xy and thrust_sp_xy_norm > 1e-8:
        #     vel_error[:2] = (
        #         vel_error[:2]
        #         - ARW_GAIN * (acc_sp[:2] - acc_sp_xy_produced)
        #     )

        vel_error = np.nan_to_num(vel_error)

        vel_int[0] += vel_error[0] * MPC_XY_VEL_I_ACC * dt
        vel_int[1] += vel_error[1] * MPC_XY_VEL_I_ACC * dt
        vel_int[2] += vel_error[2] * MPC_Z_VEL_I_ACC * dt
        vel_int[2] = np.clip(vel_int[2], -A_OF_GRAVITY, A_OF_GRAVITY)

        # ==============================
        # PX4 getAttitudeSetpoint()
        # ==============================
        att_sp_nominal = thrust_to_attitude(thr_sp, yaw_sp)
        att_sp = att_sp_nominal.copy()
        prev_att_sp_error = prev_observed_att_sp - prev_modeled_att_sp

        if online_att_models is not None:
            current_att_sp_phi = np.zeros((4, 3), dtype=float)
            modeled_att_sp = np.zeros(4)
            for axis in range(4):
                phi = np.array([att_sp_nominal[axis], prev_att_sp_error[axis], 1.0])
                current_att_sp_phi[axis] = phi
                y_hat = online_att_models[axis].predict(phi)
                modeled_att_sp[axis] = att_sp_nominal[axis] if not np.isfinite(y_hat) else y_hat
            modeled_att_sp = normalize(modeled_att_sp)
            modeled_att_sp = align_quat_sign(modeled_att_sp, att_sp_nominal)
            att_sp = modeled_att_sp
            prev_modeled_att_sp = modeled_att_sp.copy()
        elif att_sp_model is not None:
            modeled_att_sp = np.zeros(4)
            for axis in range(4):
                k_raw, k_prev_err, bias = att_sp_model[axis]
                modeled_att_sp[axis] = (
                    k_raw * att_sp_nominal[axis]
                    + k_prev_err * prev_att_sp_error[axis]
                    + bias
                )
            modeled_att_sp = normalize(modeled_att_sp)
            modeled_att_sp = align_quat_sign(modeled_att_sp, att_sp_nominal)
            att_sp = modeled_att_sp
            prev_modeled_att_sp = modeled_att_sp.copy()

        # ==============================
        # attitude setpoint -> rate setpoint
        # approximation
        # ==============================
        qe0 = (
            q[0] * att_sp[0]
            + q[1] * att_sp[1]
            + q[2] * att_sp[2]
            + q[3] * att_sp[3]
        )

        sgn = 1.0 if qe0 >= 0.0 else -1.0

        omega_sp = np.zeros(3)

        omega_sp[0] = 2.0 * MC_ROLL_P * sgn * (
            q[0] * att_sp[1]
            - q[1] * att_sp[0]
            - q[2] * att_sp[3]
            + q[3] * att_sp[2]
        )

        omega_sp[1] = 2.0 * MC_PITCH_P * sgn * (
            q[0] * att_sp[2]
            + q[1] * att_sp[3]
            - q[2] * att_sp[0]
            - q[3] * att_sp[1]
        )

        omega_sp[2] = 2.0 * MC_YAW_P * sgn * (
            q[0] * att_sp[3]
            - q[1] * att_sp[2]
            + q[2] * att_sp[1]
            - q[3] * att_sp[0]
        )

        rate_model_input = omega_sp.copy()
        if online_rate_models is not None:
            corrected_rate_sp = np.zeros(3)
            prev_rate_sp_error = prev_observed_rate_sp - prev_modeled_rate_sp
            current_rate_sp_phi = np.zeros((3, 3), dtype=float)
            for axis in range(3):
                phi = np.array([rate_model_input[axis], prev_rate_sp_error[axis], 1.0])
                current_rate_sp_phi[axis] = phi
                y_hat = online_rate_models[axis].predict(phi)
                corrected_rate_sp[axis] = rate_model_input[axis] if not np.isfinite(y_hat) else y_hat
            omega_sp = corrected_rate_sp
            prev_modeled_rate_sp = corrected_rate_sp.copy()
        elif rate_sp_model is not None:
            prev_logged_rate_sp = prev_observed_rate_sp.copy()
            if i > 0:
                logged_prev = np.array([
                    df.iloc[i - 1].get("px4_sp_roll_rate", np.nan),
                    df.iloc[i - 1].get("px4_sp_pitch_rate", np.nan),
                    df.iloc[i - 1].get("px4_sp_yaw_rate", np.nan),
                ], dtype=float)
                if np.all(np.isfinite(logged_prev)):
                    prev_logged_rate_sp = logged_prev

            prev_rate_sp_error = prev_logged_rate_sp - prev_modeled_rate_sp
            corrected_rate_sp = np.zeros(3)
            for axis in range(3):
                k_raw, k_prev_err, bias = rate_sp_model[axis]
                corrected_rate_sp[axis] = (
                    k_raw * rate_model_input[axis]
                    + k_prev_err * prev_rate_sp_error[axis]
                    + bias
                )
            omega_sp = corrected_rate_sp
            prev_modeled_rate_sp = corrected_rate_sp.copy()
            prev_observed_rate_sp = prev_logged_rate_sp.copy()

        calc["vx"][i] = vel_sp[0]
        calc["vy"][i] = vel_sp[1]
        calc["vz"][i] = vel_sp[2]

        calc["acc_x"][i] = acc_sp[0]
        calc["acc_y"][i] = acc_sp[1]
        calc["acc_z"][i] = acc_sp[2]

        calc["q0"][i] = att_sp[0]
        calc["q1"][i] = att_sp[1]
        calc["q2"][i] = att_sp[2]
        calc["q3"][i] = att_sp[3]

        calc["roll_rate"][i] = omega_sp[0]
        calc["pitch_rate"][i] = omega_sp[1]
        calc["yaw_rate"][i] = omega_sp[2]

        calc["thrust_x"][i] = thr_sp[0]
        calc["thrust_y"][i] = thr_sp[1]
        calc["thrust_z"][i] = thr_sp[2]

        if online_velocity_models is not None:
            pos_err = pos_sp - pos
            logged_vel_sp = np.array([
                row.get("px4_sp_vx", np.nan),
                row.get("px4_sp_vy", np.nan),
                row.get("px4_sp_vz", np.nan),
            ], dtype=float)
            for axis in range(3):
                if current_velocity_sp_phi is None:
                    phi = np.array([pos_err[axis], 0.0, 1.0])
                else:
                    phi = current_velocity_sp_phi[axis]
                online_velocity_models[axis].update(phi, logged_vel_sp[axis])
            if np.all(np.isfinite(logged_vel_sp)):
                prev_observed_vel_sp = logged_vel_sp.copy()
            else:
                prev_observed_vel_sp = prev_modeled_vel_sp.copy()

        if online_acc_models is not None:
            logged_acc_sp = np.array([
                row.get("px4_sp_acc_x", np.nan),
                row.get("px4_sp_acc_y", np.nan),
                row.get("px4_sp_acc_z", np.nan),
            ], dtype=float)
            for axis in range(3):
                if current_acc_sp_phi is None:
                    phi = np.array([vel_error[axis], vel_int[axis], -vel_dot[axis], 0.0, 1.0])
                else:
                    phi = current_acc_sp_phi[axis]
                online_acc_models[axis].update(phi, logged_acc_sp[axis])
            if np.all(np.isfinite(logged_acc_sp)):
                prev_observed_acc_sp = logged_acc_sp.copy()
            else:
                prev_observed_acc_sp = prev_modeled_acc_sp.copy()

        if online_att_models is not None:
            logged_att_sp = np.array([
                row.get("px4_sp_q0", np.nan),
                row.get("px4_sp_q1", np.nan),
                row.get("px4_sp_q2", np.nan),
                row.get("px4_sp_q3", np.nan),
            ], dtype=float)
            logged_att_sp = align_quat_sign(logged_att_sp, att_sp)
            for axis in range(4):
                if current_att_sp_phi is None:
                    phi = np.array([att_sp_nominal[axis], 0.0, 1.0])
                else:
                    phi = current_att_sp_phi[axis]
                online_att_models[axis].update(phi, logged_att_sp[axis])
            if np.all(np.isfinite(logged_att_sp)):
                prev_observed_att_sp = logged_att_sp.copy()
            else:
                prev_observed_att_sp = prev_modeled_att_sp.copy()

        if online_rate_models is not None:
            logged_rate_sp = np.array([
                row.get("px4_sp_roll_rate", np.nan),
                row.get("px4_sp_pitch_rate", np.nan),
                row.get("px4_sp_yaw_rate", np.nan),
            ], dtype=float)
            for axis in range(3):
                if current_rate_sp_phi is None:
                    phi = np.array([rate_model_input[axis], 0.0, 1.0])
                else:
                    phi = current_rate_sp_phi[axis]
                online_rate_models[axis].update(phi, logged_rate_sp[axis])
            if np.all(np.isfinite(logged_rate_sp)):
                prev_observed_rate_sp = logged_rate_sp.copy()
            else:
                prev_observed_rate_sp = prev_modeled_rate_sp.copy()

        prev_vel = vel.copy()

    return calc


parser = argparse.ArgumentParser(description="Compare reconstructed PX4 setpoints with logged PX4 setpoints.")
parser.add_argument("plot_name", nargs="?", default="roll_rate")
parser.add_argument("csv_path", nargs="?", default=default_csv)
parser.add_argument(
    "--source",
    choices=("auto", "u0", "target"),
    default="auto",
    help="position setpoint source used for reconstruction",
)
parser.add_argument(
    "--velocity-source",
    choices=("auto", "reconstructed", "logged"),
    default="auto",
    help="velocity setpoint used by acceleration/attitude/rate reconstruction",
)
parser.add_argument(
    "--profile",
    choices=("px4", "current", "simulation", "legacy"),
    default="px4",
    help="PX4/MPC parameter profile",
)
parser.add_argument(
    "--id-mode",
    choices=("online", "batch", "off"),
    default="online",
    help="identification mode: online is causal RLS, batch is full-log least squares",
)
parser.add_argument(
    "--no-fit",
    action="store_true",
    help="deprecated alias for --id-mode off",
)
parser.add_argument("--delay", default="auto", help="plot delay applied to reconstructed signal [s], or 'auto'")
args = parser.parse_args()

plot_name = args.plot_name
csv_path = args.csv_path
apply_param_profile(args.profile)
identification_mode = "off" if args.no_fit else args.id_mode

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

if identification_mode == "batch":
    position_source_by_row = choose_position_source_by_row(df, args.source)
    df["_viewer_position_source"] = position_source_by_row
    position_source = "row" if args.source == "auto" else args.source
else:
    position_source_by_row = None
    position_source = choose_position_source(df, args.source)

velocity_source = args.velocity_source

if identification_mode == "batch":
    fit_params_to_log(df)
    # In this script, only rate_sp_model is used for RLS/batch correction.
    # Keep velocity/acceleration/attitude as nominal PX4-like calculations.
    velocity_sp_model = None
    acc_sp_model = None
    att_sp_model = None
    if args.source == "auto":
        position_source_by_row = choose_position_source_by_row(df, args.source)
        df["_viewer_position_source"] = position_source_by_row

print(f"[INFO] csv={csv_path}")
print("[INFO] RLS target: rate setpoint only. vx/vy/vz, acc_*, q* are nominal PX4-like calculations without RLS.")
if position_source == "row":
    n_target = int(np.count_nonzero(np.asarray(position_source_by_row) == "target"))
    source_info = f"row(target={n_target}, u0={len(df) - n_target})"
else:
    source_info = position_source
print(f"[INFO] plot={plot_name}, source={source_info}, velocity_source={velocity_source}, profile={args.profile}, id_mode={identification_mode}, dt={dt:.6f}, delay={args.delay}")
print(
    "[INFO] params "
    f"MPC_XY_P={MPC_XY_P:.4g}, MPC_Z_P={MPC_Z_P:.4g}, "
    f"MPC_XY_VEL_P/I/D={MPC_XY_VEL_P_ACC:.4g}/{MPC_XY_VEL_I_ACC:.4g}/{MPC_XY_VEL_D_ACC:.4g}, "
    f"MPC_Z_VEL_P/I/D={MPC_Z_VEL_P_ACC:.4g}/{MPC_Z_VEL_I_ACC:.4g}/{MPC_Z_VEL_D_ACC:.4g}, "
    f"MC_R/P/Y={MC_ROLL_P:.4g}/{MC_PITCH_P:.4g}/{MC_YAW_P:.4g}"
)
if velocity_sp_model is not None:
    print(
        "[INFO] velocity_sp_model "
        f"x={velocity_sp_model[0]}, y={velocity_sp_model[1]}, z={velocity_sp_model[2]}"
    )
if acc_sp_model is not None:
    print(
        "[INFO] acc_sp_model "
        f"x={acc_sp_model[0]}, y={acc_sp_model[1]}, z={acc_sp_model[2]}"
    )
if att_sp_model is not None:
    print(
        "[INFO] att_sp_model "
        f"q0={att_sp_model[0]}, q1={att_sp_model[1]}, q2={att_sp_model[2]}, q3={att_sp_model[3]}"
    )
if rate_sp_model is not None:
    print(
        "[INFO] rate_sp_model "
        f"roll={rate_sp_model[0]}, pitch={rate_sp_model[1]}, yaw={rate_sp_model[2]}"
    )

calc = calc_px4_like_setpoints(df)

px4_col_map = {
    "vx": "px4_sp_vx",
    "vy": "px4_sp_vy",
    "vz": "px4_sp_vz",

    "acc_x": "px4_sp_acc_x",
    "acc_y": "px4_sp_acc_y",
    "acc_z": "px4_sp_acc_z",

    "q0": "px4_sp_q0",
    "q1": "px4_sp_q1",
    "q2": "px4_sp_q2",
    "q3": "px4_sp_q3",

    "roll_rate": "px4_sp_roll_rate",
    "pitch_rate": "px4_sp_pitch_rate",
    "yaw_rate": "px4_sp_yaw_rate",

    "thrust_x": "px4_sp_thrust_x",
    "thrust_y": "px4_sp_thrust_y",
    "thrust_z": "px4_sp_thrust_z",
}

if plot_name not in px4_col_map:
    print("[ERROR] plot_name must be one of:")
    for k in px4_col_map:
        print(" ", k)
    sys.exit(1)

px4_col = px4_col_map[plot_name]

if px4_col not in df.columns:
    print(f"[ERROR] missing column: {px4_col}")
    sys.exit(1)

t = df["t"].to_numpy()
calc_y = calc[plot_name]
px4_y = df[px4_col].to_numpy()
if np.count_nonzero(np.isfinite(px4_y)) == 0:
    print(f"[WARN] {px4_col} has no finite samples in this CSV; only calculated_{plot_name} can be inspected.")

fig, ax = plt.subplots(figsize=(12, 5))
plt.subplots_adjust(bottom=0.25, right=0.88)

if str(args.delay).lower() == "auto":
    plot_delay_sec = estimate_plot_delay(t, calc_y, px4_y)
    print(f"[INFO] selected_delay={plot_delay_sec:.3f}s")
else:
    plot_delay_sec = float(args.delay)

ax.plot(
    t + plot_delay_sec,
    calc_y,
    linewidth=2.5,
    label=f"calculated_{plot_name}_delay_{plot_delay_sec:.2f}s",
)

ax.plot(
    t,
    px4_y,
    linewidth=2.0,
    alpha=0.8,
    label=px4_col,
)

ax.set_title(plot_name)
ax.set_xlabel("time [s]")
ax.grid()
ax.legend(fontsize=14)

ax.relim()
ax.autoscale_view()

t_min = 0.0
t_max = float(t.max())

init_width = min(2.0, t_max - t_min)
window_width = [init_width]

slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
time_slider = Slider(
    slider_ax,
    "time",
    t_min,
    t_max,
    valinit=init_width / 2.0,
)

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