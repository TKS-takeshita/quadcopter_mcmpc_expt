import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CSV = "/home/ros2/ws_mcmpc/src/visualize_mcmpc/csv/offboard_control_log_step3.csv"

DT = 0.02

A_OF_GRAVITY = 9.80665
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
COM_SPOOLUP_TIME = 1.0
MPC_TKO_RAMP_T = 3.0
MPC_THR_HOVER = 0.60
MPC_THR_MIN = 0.1
MPC_THR_MAX = 0.9
MPC_THR_XY_MARGIN = 0.3
MPC_TILT_MAX = 0.78539816339
MC_ROLL_P = 4.00
MC_PITCH_P = 4.00
MC_YAW_P = 2.80
MPC_VELD_LP = 5.0

TAKEOFF_STATE_DISARMED = 1
TAKEOFF_STATE_SPOOLUP = 2
TAKEOFF_STATE_READY_FOR_TAKEOFF = 3
TAKEOFF_STATE_RAMPUP = 4
TAKEOFF_STATE_FLIGHT = 5

STATE_NAMES = [
    "e0", "e1", "e2", "e3",
    "wx", "wy", "wz",
    "x", "y", "z",
    "vx", "vy", "vz",
]
STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}


def preprocess_log(df, step_dt):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df[np.isfinite(df["control_time_s"].to_numpy(float))].reset_index(drop=True)

    t = df["control_time_s"].to_numpy(float)
    reset_indices = np.flatnonzero(np.diff(t) <= 0.0) + 1
    starts = np.r_[0, reset_indices]
    ends = np.r_[reset_indices, len(df)]

    if len(starts) > 1:
        continuous_t = np.empty_like(t)
        next_start_time = 0.0
        for start, end in zip(starts, ends):
            segment_t = t[start:end]
            continuous_t[start:end] = next_start_time + segment_t - segment_t[0]
            next_start_time = continuous_t[end - 1] + step_dt
        print(
            f"[INFO] time reset detected: stitched {len(starts)} monotonic segments "
            f"over {len(df)} rows"
        )
        df["control_time_s"] = continuous_t

    t = df["control_time_s"].to_numpy(float)
    t0 = float(t[0])
    sample_index = np.rint((t - t0) / step_dt).astype(np.int64)
    df["_sample_index"] = sample_index
    before = len(df)
    df = df.drop_duplicates("_sample_index", keep="last").sort_values("_sample_index").reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[INFO] dropped {dropped} duplicate samples after dt binning")

    df["control_time_s"] = df["_sample_index"].to_numpy(float) * step_dt
    df = df.drop(columns=["_sample_index"])
    return df


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1.0e-8:
        return v
    return v / n


def euler_to_quat(roll, pitch, yaw):
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    return normalize(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


def quat_to_euler(q):
    q0, q1, q2, q3 = normalize(q)
    roll = np.arctan2(
        2.0 * (q0 * q1 + q2 * q3),
        1.0 - 2.0 * (q1 * q1 + q2 * q2),
    )
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(
        2.0 * (q0 * q3 + q1 * q2),
        1.0 - 2.0 * (q2 * q2 + q3 * q3),
    )
    return np.array([roll, pitch, yaw])


def rotmat_to_quat(body_x, body_y, body_z):
    r00, r01, r02 = body_x[0], body_y[0], body_z[0]
    r10, r11, r12 = body_x[1], body_y[1], body_z[1]
    r20, r21, r22 = body_x[2], body_y[2], body_z[2]
    tr = r00 + r11 + r22

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s])
    elif r00 > r11 and r00 > r22:
        s = np.sqrt(1.0 + r00 - r11 - r22) * 2.0
        q = np.array([(r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s])
    elif r11 > r22:
        s = np.sqrt(1.0 + r11 - r00 - r22) * 2.0
        q = np.array([(r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s])
    else:
        s = np.sqrt(1.0 + r22 - r00 - r11) * 2.0
        q = np.array([(r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s])
    return normalize(q)


def sin_cos_simulator(x):
    if x > np.pi:
        x -= 2.0 * np.pi
    if x < -np.pi:
        x += 2.0 * np.pi
    x2 = x * x
    return (
        x * (1.0 - x2 / 6.0 + x2 * x2 / 120.0),
        1.0 - x2 / 2.0 + x2 * x2 / 24.0,
    )


def state_from_row(row):
    s = np.zeros(len(STATE_NAMES), dtype=float)
    s[0:4] = euler_to_quat(row["roll"], row["pitch"], row["yaw"])
    s[STATE_INDEX["wx"]] = row["angular_vel_x"]
    s[STATE_INDEX["wy"]] = row["angular_vel_y"]
    s[STATE_INDEX["wz"]] = row["angular_vel_z"]
    s[STATE_INDEX["x"]] = row["pos_x"]
    s[STATE_INDEX["y"]] = row["pos_y"]
    s[STATE_INDEX["z"]] = row["pos_z"]
    s[STATE_INDEX["vx"]] = row["vel_x"]
    s[STATE_INDEX["vy"]] = row["vel_y"]
    s[STATE_INDEX["vz"]] = row["vel_z"]
    return s


def input_from_row(row):
    pos_sp = np.array([
        row["target_x"] if "target_x" in row.index and np.isfinite(row["target_x"]) else row["pos_x"],
        row["target_y"] if "target_y" in row.index and np.isfinite(row["target_y"]) else row["pos_y"],
        row["target_z"] if "target_z" in row.index and np.isfinite(row["target_z"]) else row["pos_z"],
    ], dtype=float)
    yaw_sp = row["target_yaw"] if "target_yaw" in row.index and np.isfinite(row["target_yaw"]) else row["yaw"]
    return pos_sp, yaw_sp


def cu_mpc_simulator_step(
    s, pos_sp, yaw_sp, prev_vel, prev_acc, vel_int, step_dt,
    z_vel_max_up=MPC_Z_VEL_MAX_UP, thrust_min=MPC_THR_MIN, no_thrust=False,
    hover_thrust=MPC_THR_HOVER, tilt_limit=MPC_TILT_MAX,
):
    var = np.asarray(s, dtype=float).copy()
    q_prev = normalize(var[0:4])
    if np.linalg.norm(q_prev) < 1.0e-8:
        q_prev = np.array([1.0, 0.0, 0.0, 0.0])
    var[0:4] = q_prev

    x_ref, y_ref, z_ref = np.asarray(pos_sp, dtype=float)
    yaw_ref = float(yaw_sp)

    vel_prev_state = var[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]].copy()
    vel_dot = (vel_prev_state - prev_vel) / step_dt
    if MPC_VELD_LP > 1.0e-6:
        vel_dot_alpha = step_dt / (step_dt + 1.0 / (2.0 * np.pi * MPC_VELD_LP))
    else:
        vel_dot_alpha = 1.0
    vel_dot_lpf = prev_acc + vel_dot_alpha * (vel_dot - prev_acc)

    if no_thrust or not np.all(np.isfinite([x_ref, y_ref, z_ref])):
        acc_setpoint = np.array([0.0, 0.0, 100.0])
        thrust_setpoint = np.zeros(3)
        return var.copy(), vel_dot_lpf, np.zeros(3), {
            "vel_sp": np.full(3, np.nan),
            "acc_sp": acc_setpoint,
            "att_sp": np.array([1.0, 0.0, 0.0, 0.0]),
            "rate_sp": np.zeros(3),
            "thr_sp": thrust_setpoint,
        }

    vel_setpoint_position = np.array([
        MPC_XY_P * (x_ref - var[STATE_INDEX["x"]]),
        MPC_XY_P * (y_ref - var[STATE_INDEX["y"]]),
        MPC_Z_P * (z_ref - var[STATE_INDEX["z"]]),
    ])
    vel_setpoint = vel_setpoint_position.copy()
    vel_xy_norm = np.sqrt(vel_setpoint[0] ** 2 + vel_setpoint[1] ** 2)
    if vel_xy_norm > MPC_XY_VEL_MAX and vel_xy_norm > 1.0e-8:
        vel_setpoint[:2] = vel_setpoint[:2] / vel_xy_norm * MPC_XY_VEL_MAX
    z_vel_min = -z_vel_max_up
    if vel_setpoint[2] < z_vel_min:
        vel_setpoint[2] = z_vel_min
    elif vel_setpoint[2] > MPC_Z_VEL_MAX_DOWN:
        vel_setpoint[2] = MPC_Z_VEL_MAX_DOWN

    vel_error = vel_setpoint - vel_prev_state

    acc_setpoint = np.array([
        MPC_XY_VEL_P_ACC * vel_error[0] + vel_int[0] - MPC_XY_VEL_D_ACC * vel_dot_lpf[0],
        MPC_XY_VEL_P_ACC * vel_error[1] + vel_int[1] - MPC_XY_VEL_D_ACC * vel_dot_lpf[1],
        MPC_Z_VEL_P_ACC * vel_error[2] + vel_int[2] - MPC_Z_VEL_D_ACC * vel_dot_lpf[2],
    ])

    body_z = np.array([-acc_setpoint[0], -acc_setpoint[1], A_OF_GRAVITY - acc_setpoint[2]])
    body_z = normalize(body_z)
    if np.linalg.norm(body_z) < 1.0e-8:
        body_z = np.array([0.0, 0.0, 1.0])

    tilt_angle = np.arccos(np.clip(body_z[2], -1.0, 1.0))
    if tilt_angle > tilt_limit:
        rejection = np.array([body_z[0], body_z[1], 0.0])
        rejection_norm = np.linalg.norm(rejection)
        if rejection_norm < 1.0e-8:
            rejection = np.array([1.0, 0.0, 0.0])
            rejection_norm = 1.0
        rejection = rejection / rejection_norm
        body_z = np.array([
            np.sin(tilt_limit) * rejection[0],
            np.sin(tilt_limit) * rejection[1],
            np.cos(tilt_limit),
        ])

    thrust_ned_z = acc_setpoint[2] * (hover_thrust / A_OF_GRAVITY) - hover_thrust
    cos_ned_body = body_z[2] if abs(body_z[2]) >= 1.0e-6 else 1.0e-6
    collective_thrust = min(thrust_ned_z / cos_ned_body, -thrust_min)
    thrust_setpoint = body_z * collective_thrust

    vel_error_for_int = vel_error.copy()
    if (thrust_setpoint[2] >= -thrust_min and vel_error_for_int[2] >= 0.0) or \
       (thrust_setpoint[2] <= -MPC_THR_MAX and vel_error_for_int[2] <= 0.0):
        vel_error_for_int[2] = 0.0

    thrust_sp_xy_norm = np.sqrt(thrust_setpoint[0] ** 2 + thrust_setpoint[1] ** 2)
    thrust_max_squared = MPC_THR_MAX * MPC_THR_MAX
    allocated_horizontal_thrust = min(thrust_sp_xy_norm, MPC_THR_XY_MARGIN)
    thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust * allocated_horizontal_thrust
    thrust_setpoint[2] = max(thrust_setpoint[2], -np.sqrt(max(0.0, thrust_z_max_squared)))

    thrust_max_xy_squared = thrust_max_squared - thrust_setpoint[2] * thrust_setpoint[2]
    thrust_max_xy = np.sqrt(max(0.0, thrust_max_xy_squared))
    if thrust_sp_xy_norm > thrust_max_xy and thrust_sp_xy_norm > 1.0e-8:
        thrust_setpoint[:2] = thrust_setpoint[:2] / thrust_sp_xy_norm * thrust_max_xy

    acc_sp_xy_produced = thrust_setpoint[:2] * (A_OF_GRAVITY / hover_thrust)
    if np.dot(acc_setpoint[:2], acc_setpoint[:2]) > np.dot(acc_sp_xy_produced, acc_sp_xy_produced):
        arw_gain = 2.0 / MPC_XY_VEL_P_ACC
        vel_error_for_int[:2] -= arw_gain * (acc_setpoint[:2] - acc_sp_xy_produced)
    vel_error_for_int[~np.isfinite(vel_error_for_int)] = 0.0

    att_body_z = normalize(-thrust_setpoint)
    if np.linalg.norm(att_body_z) < 1.0e-8:
        att_body_z = np.array([0.0, 0.0, 1.0])
    sy, cy = sin_cos_simulator(yaw_ref)
    y_c = np.array([-sy, cy, 0.0])
    body_x = np.cross(y_c, att_body_z)
    if att_body_z[2] < 0.0:
        body_x = -body_x
    if abs(att_body_z[2]) < 1.0e-6:
        body_x = np.array([0.0, 0.0, 1.0])
    body_x = normalize(body_x)
    body_y = np.cross(att_body_z, body_x)
    att_setpoint = rotmat_to_quat(body_x, body_y, att_body_z)

    qe0 = np.dot(var[0:4], att_setpoint)
    sgn = 1.0 if qe0 >= 0.0 else -1.0
    omega_setpoint = np.array([
        2.0 * MC_ROLL_P * sgn * (var[0] * att_setpoint[1] - var[1] * att_setpoint[0] - var[2] * att_setpoint[3] + var[3] * att_setpoint[2]),
        2.0 * MC_PITCH_P * sgn * (var[0] * att_setpoint[2] + var[1] * att_setpoint[3] - var[2] * att_setpoint[0] - var[3] * att_setpoint[1]),
        2.0 * MC_YAW_P * sgn * (var[0] * att_setpoint[3] - var[1] * att_setpoint[2] + var[2] * att_setpoint[1] - var[3] * att_setpoint[0]),
    ])

    v_prev = vel_prev_state.copy()
    w_prev = var[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]].copy()
    next_var = var.copy()
    next_var[STATE_INDEX["x"]] += v_prev[0] * step_dt
    next_var[STATE_INDEX["y"]] += v_prev[1] * step_dt
    next_var[STATE_INDEX["z"]] += v_prev[2] * step_dt
    next_var[STATE_INDEX["vx"]] = v_prev[0] + acc_setpoint[0] * step_dt
    next_var[STATE_INDEX["vy"]] = v_prev[1] + acc_setpoint[1] * step_dt
    next_var[STATE_INDEX["vz"]] = v_prev[2] + acc_setpoint[2] * step_dt

    q_dot = np.array([
        -0.5 * (q_prev[1] * w_prev[0] + q_prev[2] * w_prev[1] + q_prev[3] * w_prev[2]),
        0.5 * (q_prev[0] * w_prev[0] + q_prev[2] * w_prev[2] - q_prev[3] * w_prev[1]),
        0.5 * (q_prev[0] * w_prev[1] - q_prev[1] * w_prev[2] + q_prev[3] * w_prev[0]),
        0.5 * (q_prev[0] * w_prev[2] + q_prev[1] * w_prev[1] - q_prev[2] * w_prev[0]),
    ])
    next_var[0:4] = normalize(q_prev + q_dot * step_dt)
    next_var[STATE_INDEX["wx"]] = omega_setpoint[0]
    next_var[STATE_INDEX["wy"]] = omega_setpoint[1]
    next_var[STATE_INDEX["wz"]] = omega_setpoint[2]

    vel_int_next = vel_int.copy()
    vel_int_next[0] += vel_error_for_int[0] * MPC_XY_VEL_I_ACC * step_dt
    vel_int_next[1] += vel_error_for_int[1] * MPC_XY_VEL_I_ACC * step_dt
    vel_int_next[2] += vel_error_for_int[2] * MPC_Z_VEL_I_ACC * step_dt
    vel_int_next[2] = np.clip(vel_int_next[2], -A_OF_GRAVITY, A_OF_GRAVITY)

    return next_var, vel_dot_lpf, vel_int_next, {
        "vel_sp": vel_setpoint,
        "acc_sp": acc_setpoint,
        "att_sp": att_setpoint,
        "rate_sp": omega_setpoint,
        "thr_sp": thrust_setpoint,
    }


def build_history(df, step_dt):
    n = len(df)
    pred_next = np.full((n, len(STATE_NAMES)), np.nan)
    actual = np.full((n, len(STATE_NAMES)), np.nan)
    actual_next = np.full((n, len(STATE_NAMES)), np.nan)
    calc = {
        "vel_sp": np.full((n, 3), np.nan),
        "acc_sp": np.full((n, 3), np.nan),
        "att_sp": np.full((n, 4), np.nan),
        "rate_sp": np.full((n, 3), np.nan),
        "thr_sp": np.full((n, 3), np.nan),
    }

    prev_vel = df.loc[0, ["vel_x", "vel_y", "vel_z"]].to_numpy(float)
    prev_acc = np.zeros(3)
    vel_int = np.zeros(3)
    hover_thrust = MPC_THR_HOVER
    last_hover_thrust_log = np.nan
    last_acc_sp = np.zeros(3)
    takeoff_ramp_vz_init = -A_OF_GRAVITY / max(MPC_Z_VEL_P_ACC, 0.01)
    takeoff_ramp_progress = 0.0
    takeoff_state = TAKEOFF_STATE_SPOOLUP
    spoolup_elapsed = 0.0

    for i in range(n):
        row = df.iloc[i]
        s = state_from_row(row)
        actual[i] = s
        if i + 1 < n:
            actual_next[i] = state_from_row(df.iloc[i + 1])

        pos_sp, yaw_sp = input_from_row(row)
        if "hover_thrust" in row.index and "hover_thrust_valid" in row.index:
            hover_thrust_new = float(row["hover_thrust"])
            hover_thrust_valid = bool(np.isfinite(row["hover_thrust_valid"]) and row["hover_thrust_valid"] != 0.0)
            hover_thrust_updated = (
                not np.isfinite(last_hover_thrust_log)
                or abs(hover_thrust_new - last_hover_thrust_log) > 1.0e-7
            )
            if hover_thrust_valid and hover_thrust_updated and np.isfinite(hover_thrust_new) and hover_thrust_new > 1.0e-6:
                previous_hover_thrust = hover_thrust
                hover_thrust = hover_thrust_new
                vel_int[2] += (
                    (last_acc_sp[2] - A_OF_GRAVITY) * previous_hover_thrust / hover_thrust
                    + A_OF_GRAVITY - last_acc_sp[2]
                )
                last_hover_thrust_log = hover_thrust_new

        want_takeoff = np.isfinite(pos_sp[2]) and pos_sp[2] < s[STATE_INDEX["z"]]
        landed = bool(row["landed"]) if "landed" in row.index and np.isfinite(row["landed"]) else False
        maybe_landed = (
            bool(row["maybe_landed"]) if "maybe_landed" in row.index and np.isfinite(row["maybe_landed"]) else False
        )
        ground_contact = (
            bool(row["ground_contact"]) if "ground_contact" in row.index and np.isfinite(row["ground_contact"]) else False
        )
        tilt_limit = (
            float(row["takeoff_tilt_limit"])
            if "takeoff_tilt_limit" in row.index and np.isfinite(row["takeoff_tilt_limit"])
            else MPC_TILT_MAX
        )

        if "takeoff_state" in row.index and np.isfinite(row["takeoff_state"]):
            takeoff_state = int(row["takeoff_state"])
        else:
            if takeoff_state == TAKEOFF_STATE_DISARMED:
                takeoff_state = TAKEOFF_STATE_SPOOLUP

            if takeoff_state == TAKEOFF_STATE_SPOOLUP:
                spoolup_elapsed += step_dt
                if spoolup_elapsed >= COM_SPOOLUP_TIME:
                    takeoff_state = TAKEOFF_STATE_READY_FOR_TAKEOFF

            if takeoff_state == TAKEOFF_STATE_READY_FOR_TAKEOFF and want_takeoff:
                takeoff_state = TAKEOFF_STATE_RAMPUP
                takeoff_ramp_progress = 0.0

            if takeoff_state == TAKEOFF_STATE_RAMPUP and takeoff_ramp_progress >= 1.0:
                takeoff_state = TAKEOFF_STATE_FLIGHT

            if takeoff_state == TAKEOFF_STATE_FLIGHT and (landed or maybe_landed):
                takeoff_state = TAKEOFF_STATE_READY_FOR_TAKEOFF

        not_taken_off = takeoff_state < TAKEOFF_STATE_RAMPUP
        flying = takeoff_state >= TAKEOFF_STATE_FLIGHT
        flying_but_ground_contact = flying and (ground_contact or maybe_landed)

        if takeoff_state == TAKEOFF_STATE_RAMPUP:
            takeoff_ramp_progress = min(1.0, takeoff_ramp_progress + step_dt / MPC_TKO_RAMP_T)
            z_vel_max_up = takeoff_ramp_vz_init + takeoff_ramp_progress * (MPC_Z_VEL_MAX_UP - takeoff_ramp_vz_init)
        elif takeoff_state >= TAKEOFF_STATE_FLIGHT:
            z_vel_max_up = MPC_Z_VEL_MAX_UP
        else:
            z_vel_max_up = takeoff_ramp_vz_init

        thrust_min = MPC_THR_MIN if flying else 0.0
        if not flying:
            hover_thrust = MPC_THR_HOVER

        s_next, prev_acc, vel_int, debug = cu_mpc_simulator_step(
            s, pos_sp, yaw_sp, prev_vel, prev_acc, vel_int, step_dt,
            z_vel_max_up=z_vel_max_up, thrust_min=thrust_min,
            no_thrust=(not_taken_off or flying_but_ground_contact),
            hover_thrust=hover_thrust, tilt_limit=tilt_limit,
        )
        last_acc_sp = debug["acc_sp"].copy()
        pred_next[i] = s_next
        for key in calc:
            calc[key][i] = debug[key]
        prev_vel = s[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]].copy()

    return actual, actual_next, pred_next, calc


def finite_plot_mask(t, *values):
    mask = np.isfinite(t)
    for value in values:
        mask &= np.isfinite(value)
    return mask


def non_idle_plot_mask(df):
    if "phase" not in df.columns:
        return np.ones(len(df), dtype=bool)
    phase = df["phase"].astype(str).str.strip()
    return phase.ne("Idle").to_numpy()


def setpoint_legend_base(key):
    bases = {
        "vel_sp": "velocity_setpoint",
        "acc_sp": "acceleration_setpoint",
        "att_sp": "attitude_setpoint",
        "rate_sp": "omega_setpoint",
        "thr_sp": "thrust_setpoint",
    }
    return bases.get(key, key)


def attach_interactive_navigation(fig, ax, base_scale=1.2):
    drag = {"pressed": False, "x": None, "y": None, "xlim": None, "ylim": None}

    def on_scroll(event):
        if event.inaxes != ax:
            return

        scale = 1.0 / base_scale if event.button == "up" else base_scale

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_center = event.xdata if event.xdata is not None else 0.5 * (x_min + x_max)
        y_center = event.ydata if event.ydata is not None else 0.5 * (y_min + y_max)

        new_x_half = 0.5 * (x_max - x_min) * scale
        new_y_half = 0.5 * (y_max - y_min) * scale
        ax.set_xlim(x_center - new_x_half, x_center + new_x_half)
        ax.set_ylim(y_center - new_y_half, y_center + new_y_half)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        drag["pressed"] = True
        drag["x"] = event.xdata
        drag["y"] = event.ydata
        drag["xlim"] = ax.get_xlim()
        drag["ylim"] = ax.get_ylim()

    def on_motion(event):
        if not drag["pressed"] or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None or drag["x"] is None or drag["y"] is None:
            return

        dx = event.xdata - drag["x"]
        dy = event.ydata - drag["y"]
        x_min, x_max = drag["xlim"]
        y_min, y_max = drag["ylim"]
        ax.set_xlim(x_min - dx, x_max - dx)
        ax.set_ylim(y_min - dy, y_max - dy)
        fig.canvas.draw_idle()

    def on_release(event):
        if event.button == 1:
            drag["pressed"] = False

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)


def main():
    parser = argparse.ArgumentParser(description="Compare offboard log setpoints and one-step dynamics against mpc_simulator.cu-style prediction.")
    parser.add_argument("state", nargs="?", default="wx")
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    parser.add_argument("--dt", type=float, default=DT)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = preprocess_log(df, args.dt)
    t = df["control_time_s"].to_numpy(float)
    t_diff = np.diff(t)
    finite_dt = t_diff[np.isfinite(t_diff) & (t_diff > 0.0)]
    csv_dt = float(np.median(finite_dt)) if len(finite_dt) > 0 else args.dt

    actual, actual_next, pred_next, calc = build_history(df, args.dt)
    active_plot_mask = non_idle_plot_mask(df)
    if np.any(active_plot_mask):
        plot_t = t - t[np.flatnonzero(active_plot_mask)[0]]
    else:
        plot_t = t.copy()

    state_aliases = {
        "e0": "q0", "e1": "q1", "e2": "q2", "e3": "q3",
        "pos_x": "x", "pos_y": "y", "pos_z": "z",
        "vel_x": "vx", "vel_y": "vy", "vel_z": "vz",
        "angular_vel_x": "wx", "angular_vel_y": "wy", "angular_vel_z": "wz",
        "roll_rate": "wx", "pitch_rate": "wy", "yaw_rate": "wz",
        "local_sp_vx": "vel_sp_x", "local_sp_vy": "vel_sp_y", "local_sp_vz": "vel_sp_z",
        "local_sp_ax": "acc_sp_x", "local_sp_ay": "acc_sp_y", "local_sp_az": "acc_sp_z",
    }
    state = state_aliases.get(args.state, args.state)

    dynamics_sources = {
        "q0": STATE_INDEX["e0"], "q1": STATE_INDEX["e1"],
        "q2": STATE_INDEX["e2"], "q3": STATE_INDEX["e3"],
        "x": STATE_INDEX["x"], "y": STATE_INDEX["y"], "z": STATE_INDEX["z"],
        "vx": STATE_INDEX["vx"], "vy": STATE_INDEX["vy"], "vz": STATE_INDEX["vz"],
        "wx": STATE_INDEX["wx"], "wy": STATE_INDEX["wy"], "wz": STATE_INDEX["wz"],
    }
    euler_sources = {"roll": 0, "pitch": 1, "yaw": 2}
    setpoint_sources = {
        "vel_sp_x": ("vel_sp", 0, "local_sp_vx"),
        "vel_sp_y": ("vel_sp", 1, "local_sp_vy"),
        "vel_sp_z": ("vel_sp", 2, "local_sp_vz"),
        "acc_sp_x": ("acc_sp", 0, "local_sp_ax"),
        "acc_sp_y": ("acc_sp", 1, "local_sp_ay"),
        "acc_sp_z": ("acc_sp", 2, "local_sp_az"),
        "att_sp_qw": ("att_sp", 0, "att_sp_qw"),
        "att_sp_qx": ("att_sp", 1, "att_sp_qx"),
        "att_sp_qy": ("att_sp", 2, "att_sp_qy"),
        "att_sp_qz": ("att_sp", 3, "att_sp_qz"),
        "rate_sp_roll": ("rate_sp", 0, "rate_sp_roll"),
        "rate_sp_pitch": ("rate_sp", 1, "rate_sp_pitch"),
        "rate_sp_yaw": ("rate_sp", 2, "rate_sp_yaw"),
        "thr_sp_x": ("thr_sp", 0, "rate_sp_thrust_x"),
        "thr_sp_y": ("thr_sp", 1, "rate_sp_thrust_y"),
        "thr_sp_z": ("thr_sp", 2, "rate_sp_thrust_z"),
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.subplots_adjust(bottom=0.18)
    plot_title = state

    if state in setpoint_sources:
        key, axis, log_col = setpoint_sources[state]
        legend_base = setpoint_legend_base(key)
        plot_title = legend_base
        model_mask = finite_plot_mask(t, calc[key][:, axis]) & active_plot_mask
        ax.plot(plot_t[model_mask], calc[key][model_mask, axis], color="tab:cyan", linewidth=2.0, label=f"{legend_base}_dynamics")
        if log_col in df.columns:
            log_value = df[log_col].to_numpy(float)
            log_mask = finite_plot_mask(t, log_value) & active_plot_mask
            ax.plot(
                plot_t[log_mask], log_value[log_mask],
                color="tab:orange", linestyle="--", linewidth=2.0, zorder=3,
                label=f"{legend_base}_px4",
            )
        ax.set_ylabel("setpoint")
    elif state in dynamics_sources:
        idx = dynamics_sources[state]
        model_delta = pred_next[:, idx] - actual[:, idx]
        log_delta = actual_next[:, idx] - actual[:, idx]
        mask = finite_plot_mask(t, model_delta, log_delta) & active_plot_mask
        plot_title = f"d{state}"
        ax.plot(plot_t[mask], model_delta[mask], color="tab:cyan", linewidth=2.0, label=f"d{state}_model")
        ax.plot(plot_t[mask], log_delta[mask], color="tab:orange", linewidth=2.0, label=f"d{state}_dynamics")
        ax.set_ylabel("1step delta")
    elif state in euler_sources:
        axis = euler_sources[state]
        pred_euler = np.array([quat_to_euler(q) for q in pred_next[:, 0:4]])
        actual_euler = np.array([quat_to_euler(q) for q in actual[:, 0:4]])
        actual_next_euler = np.array([quat_to_euler(q) if np.all(np.isfinite(q)) else [np.nan, np.nan, np.nan] for q in actual_next[:, 0:4]])
        model_delta = pred_euler[:, axis] - actual_euler[:, axis]
        log_delta = actual_next_euler[:, axis] - actual_euler[:, axis]
        mask = finite_plot_mask(t, model_delta, log_delta) & active_plot_mask
        plot_title = f"d{state}"
        ax.plot(plot_t[mask], model_delta[mask], color="tab:cyan", linewidth=2.0, label=f"d{state}_model")
        ax.plot(plot_t[mask], log_delta[mask], color="tab:orange", linewidth=2.0, label=f"d{state}_dynamics")
        ax.set_ylabel("1step delta")
    else:
        valid = sorted(set(dynamics_sources) | set(euler_sources) | set(setpoint_sources))
        print("[ERROR] state must be one of:")
        for name in valid:
            print(" ", name)
        sys.exit(1)

    print(f"[INFO] csv={args.csv_path}")
    print(f"[INFO] state={state}, model_dt={args.dt:.6f}, csv_dt={csv_dt:.6f}")
    print(f"[INFO] plot samples: {int(np.count_nonzero(active_plot_mask))}/{len(active_plot_mask)} non-Idle")
    ax.set_title(plot_title)
    ax.set_xlabel("time [s]")
    ax.grid()
    ax.legend(fontsize=12)
    ax.relim()
    ax.autoscale_view()
    attach_interactive_navigation(fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
