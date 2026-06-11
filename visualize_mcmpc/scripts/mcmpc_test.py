#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


STATE_NAMES = [
    "e0", "e1", "e2", "e3",
    "wx", "wy", "wz",
    "x", "y", "z",
    "vx", "vy", "vz",
]
IDX = {name: i for i, name in enumerate(STATE_NAMES)}

HORIZON = 75
CONTROL_PERIOD = 0.05
N_OF_SAMPLES = 128 * 64
N_OF_ELITE = 50
SAME_CYCLE_ITERATIONS = 5

A_OF_GRAVITY = 9.80665
MPC_VELD_LP = 5.0
MPC_XY_VEL_MAX = 12.0
MPC_Z_VEL_MAX_UP = 3.0
MPC_Z_VEL_MAX_DOWN = 1.0
MPC_THR_HOVER = 0.65
MPC_THR_MIN = 0.1
MPC_THR_MAX = 0.9
MPC_THR_XY_MARGIN = 0.3
MPC_TILT_MAX = np.deg2rad(45.0)
MC_ROLL_P = 3.300
MC_PITCH_P = 3.300
MC_YAW_P = 2.80

MODEL_VEL_A = np.array([1.0096476, 1.0116026, 0.99311937], dtype=np.float64)
MODEL_VEL_B = np.array([0.012726781, 0.013022681, 0.00016697574], dtype=np.float64)
MODEL_W_C = np.array([0.81389003, 0.81882324, 0.99194815], dtype=np.float64)
MODEL_W_D = np.array([0.12783148, 0.10071088, -6.9079037e-05], dtype=np.float64)
MODEL_POS_ALPHA = np.array([0.46913642, 0.47037031, 0.48688093], dtype=np.float64)
MODEL_ATT_BETA = 0.41367774

SIGMA = np.array([0.5, 0.5, 0.01, 0.001], dtype=np.float64)
SIGMA_DECAY = 0.5

COST_Q = {
    "x": 1.0,
    "y": 1.0,
    "z": 1.0,
    "vx": 5.0,
    "vy": 5.0,
    "vz": 5.0,
    "e1": 1.0,
    "e2": 1.0,
    "e3": 1.0,
    "wx": 5.0,
    "wy": 5.0,
    "wz": 5.0,
    "zi": 0.0,
}
COST_R = {"x": 0.01, "y": 0.01, "z": 0.01, "yaw": 0.01}


@dataclass
class Params:
    mpc_xy_p: float
    mpc_z_p: float = 1.0
    mpc_xy_vel_p_acc: float = 1.8
    mpc_xy_vel_i_acc: float = 0.4
    mpc_xy_vel_d_acc: float = 0.2
    mpc_z_vel_p_acc: float = 4.0
    mpc_z_vel_i_acc: float = 2.0
    mpc_z_vel_d_acc: float = 0.0


@dataclass
class RolloutStep:
    t: float
    state: np.ndarray
    average_input: np.ndarray
    best_input: np.ndarray
    cost: float


@dataclass
class DebugResult:
    samples: np.ndarray
    costs: np.ndarray
    elite_indices: np.ndarray
    elite_samples: np.ndarray
    elite_predictions: np.ndarray
    best_input: np.ndarray
    best_prediction: np.ndarray


def apply_param_profile(profile: str) -> float:
    global MPC_THR_HOVER, MC_ROLL_P, MC_PITCH_P, MC_YAW_P

    if profile == "simulation":
        MPC_THR_HOVER = 0.60
        MC_ROLL_P = 4.00
        MC_PITCH_P = 4.00
        MC_YAW_P = 2.80
        return 0.30
    if profile == "current":
        MPC_THR_HOVER = 0.65
        MC_ROLL_P = 4.00
        MC_PITCH_P = 6.80
        MC_YAW_P = 2.80
        return 0.95
    if profile == "legacy":
        MPC_THR_HOVER = 0.65
        MC_ROLL_P = 4.00
        MC_PITCH_P = 2.00
        MC_YAW_P = 2.80
        return 0.30

    MPC_THR_HOVER = 0.65
    MC_ROLL_P = 4.00
    MC_PITCH_P = 2.00
    MC_YAW_P = 2.80
    return 0.95


def parse_vec3(text: str) -> np.ndarray:
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected comma separated x,y,z")
    return np.array(parts, dtype=np.float64)


def parse_vec4(text: str) -> np.ndarray:
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected comma separated x,y,z,yaw")
    return np.array(parts, dtype=np.float64)


def make_state(position: np.ndarray, velocity: np.ndarray, yaw: float) -> np.ndarray:
    state = np.zeros(len(STATE_NAMES), dtype=np.float64)
    state[IDX["e0"]] = np.cos(0.5 * yaw)
    state[IDX["e3"]] = np.sin(0.5 * yaw)
    state[IDX["x"]:IDX["z"] + 1] = position
    state[IDX["vx"]:IDX["vz"] + 1] = velocity
    return state


def make_target(target_xyz_yaw: np.ndarray) -> dict:
    yaw = target_xyz_yaw[3]
    return {
        "e0": np.cos(0.5 * yaw),
        "e1": 0.0,
        "e2": 0.0,
        "e3": np.sin(0.5 * yaw),
        "wx": 0.0,
        "wy": 0.0,
        "wz": 0.0,
        "x": target_xyz_yaw[0],
        "y": target_xyz_yaw[1],
        "z": target_xyz_yaw[2],
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "yaw": yaw,
    }


def state_yaw(state: np.ndarray) -> float:
    q0, q1, q2, q3 = state[IDX["e0"]:IDX["e3"] + 1]
    return float(np.arctan2(
        2.0 * (q0 * q3 + q1 * q2),
        q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
    ))


def make_initial_average_input(initial_state: np.ndarray) -> np.ndarray:
    initial_input = np.array([
        initial_state[IDX["x"]],
        initial_state[IDX["y"]],
        initial_state[IDX["z"]],
        state_yaw(initial_state),
    ], dtype=np.float64)
    return np.tile(initial_input, (HORIZON, 1))


def shift_input(input_seq: np.ndarray) -> np.ndarray:
    shifted = input_seq.copy()
    shifted[:-1] = shifted[1:]
    return shifted


def weighted_average(elite_samples: np.ndarray, elite_costs: np.ndarray) -> np.ndarray:
    lam = float(np.mean(elite_costs))
    if not np.isfinite(lam) or abs(lam) < 1e-9:
        return np.mean(elite_samples, axis=0)

    weights = np.exp(-elite_costs / lam)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return np.mean(elite_samples, axis=0)

    return np.tensordot(weights / weight_sum, elite_samples, axes=(0, 0))


def generate_samples(
    average_input: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    sigma: np.ndarray = SIGMA,
) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=(n_samples, HORIZON, 4))
    return average_input[None, :, :] + noise


def normalize_batch(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=v.copy(), where=n > 1e-8)


def quat_mul_batch(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.column_stack([
        q[:, 0] * r[:, 0] - q[:, 1] * r[:, 1] - q[:, 2] * r[:, 2] - q[:, 3] * r[:, 3],
        q[:, 0] * r[:, 1] + q[:, 1] * r[:, 0] + q[:, 2] * r[:, 3] - q[:, 3] * r[:, 2],
        q[:, 0] * r[:, 2] - q[:, 1] * r[:, 3] + q[:, 2] * r[:, 0] + q[:, 3] * r[:, 1],
        q[:, 0] * r[:, 3] + q[:, 1] * r[:, 2] - q[:, 2] * r[:, 1] + q[:, 3] * r[:, 0],
    ])


def rotmat_to_quat_batch(body_x: np.ndarray, body_y: np.ndarray, body_z: np.ndarray) -> np.ndarray:
    r00, r01, r02 = body_x[:, 0], body_y[:, 0], body_z[:, 0]
    r10, r11, r12 = body_x[:, 1], body_y[:, 1], body_z[:, 1]
    r20, r21, r22 = body_x[:, 2], body_y[:, 2], body_z[:, 2]
    tr = r00 + r11 + r22
    q = np.zeros((body_x.shape[0], 4), dtype=np.float64)

    mask = tr > 0.0
    s = np.sqrt(np.maximum(tr[mask] + 1.0, 0.0)) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (r21[mask] - r12[mask]) / s
    q[mask, 2] = (r02[mask] - r20[mask]) / s
    q[mask, 3] = (r10[mask] - r01[mask]) / s

    mask_x = (~mask) & (r00 > r11) & (r00 > r22)
    s = np.sqrt(np.maximum(1.0 + r00[mask_x] - r11[mask_x] - r22[mask_x], 0.0)) * 2.0
    q[mask_x, 0] = (r21[mask_x] - r12[mask_x]) / s
    q[mask_x, 1] = 0.25 * s
    q[mask_x, 2] = (r01[mask_x] + r10[mask_x]) / s
    q[mask_x, 3] = (r02[mask_x] + r20[mask_x]) / s

    mask_y = (~mask) & (~mask_x) & (r11 > r22)
    s = np.sqrt(np.maximum(1.0 + r11[mask_y] - r00[mask_y] - r22[mask_y], 0.0)) * 2.0
    q[mask_y, 0] = (r02[mask_y] - r20[mask_y]) / s
    q[mask_y, 1] = (r01[mask_y] + r10[mask_y]) / s
    q[mask_y, 2] = 0.25 * s
    q[mask_y, 3] = (r12[mask_y] + r21[mask_y]) / s

    mask_z = (~mask) & (~mask_x) & (~mask_y)
    s = np.sqrt(np.maximum(1.0 + r22[mask_z] - r00[mask_z] - r11[mask_z], 0.0)) * 2.0
    q[mask_z, 0] = (r10[mask_z] - r01[mask_z]) / s
    q[mask_z, 1] = (r02[mask_z] + r20[mask_z]) / s
    q[mask_z, 2] = (r12[mask_z] + r21[mask_z]) / s
    q[mask_z, 3] = 0.25 * s
    return normalize_batch(q)


def thrust_to_attitude_simulator_batch(thr_sp: np.ndarray, yaw_sp: np.ndarray) -> np.ndarray:
    body_z = -thr_sp.copy()
    body_z_norm = np.linalg.norm(body_z, axis=1)
    fallback = body_z_norm < 1e-8
    body_z[fallback] = np.array([0.0, 0.0, 1.0])
    body_z = normalize_batch(body_z)

    yaw = yaw_sp.copy()
    yaw = np.where(yaw > np.pi, yaw - 2.0 * np.pi, yaw)
    yaw = np.where(yaw < -np.pi, yaw + 2.0 * np.pi, yaw)
    yaw2 = yaw * yaw
    sy = yaw * (1.0 - yaw2 / 6.0 + yaw2 * yaw2 / 120.0)
    cy = 1.0 - yaw2 / 2.0 + yaw2 * yaw2 / 24.0
    y_c = np.column_stack([-sy, cy, np.zeros_like(yaw)])

    body_x = np.cross(y_c, body_z)
    body_x[body_z[:, 2] < 0.0] *= -1.0
    body_x[np.abs(body_z[:, 2]) < 1e-6] = np.array([0.0, 0.0, 1.0])
    body_x = normalize_batch(body_x)
    body_y = np.cross(body_z, body_x)
    return rotmat_to_quat_batch(body_x, body_y, body_z)


def simulate_batch(
    initial_state: np.ndarray,
    inputs: np.ndarray,
    target: dict,
    params: Params,
    return_predictions: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    n_samples = inputs.shape[0]
    q = np.tile(initial_state[IDX["e0"]:IDX["e3"] + 1], (n_samples, 1)).astype(np.float64)
    q = normalize_batch(q)
    zero_q = np.linalg.norm(q, axis=1) < 1e-8
    q[zero_q] = np.array([1.0, 0.0, 0.0, 0.0])
    wx = np.full(n_samples, initial_state[IDX["wx"]], dtype=np.float64)
    wy = np.full(n_samples, initial_state[IDX["wy"]], dtype=np.float64)
    wz = np.full(n_samples, initial_state[IDX["wz"]], dtype=np.float64)
    x = np.full(n_samples, initial_state[IDX["x"]], dtype=np.float64)
    y = np.full(n_samples, initial_state[IDX["y"]], dtype=np.float64)
    z = np.full(n_samples, initial_state[IDX["z"]], dtype=np.float64)
    vx = np.full(n_samples, initial_state[IDX["vx"]], dtype=np.float64)
    vy = np.full(n_samples, initial_state[IDX["vy"]], dtype=np.float64)
    vz = np.full(n_samples, initial_state[IDX["vz"]], dtype=np.float64)

    prev_vx = np.full(n_samples, initial_state[IDX["vx"]], dtype=np.float64)
    prev_vy = np.full(n_samples, initial_state[IDX["vy"]], dtype=np.float64)
    prev_vz = np.full(n_samples, initial_state[IDX["vz"]], dtype=np.float64)
    prev_acc_x = np.zeros(n_samples, dtype=np.float64)
    prev_acc_y = np.zeros(n_samples, dtype=np.float64)
    prev_acc_z = np.zeros(n_samples, dtype=np.float64)
    vel_int_x = np.zeros(n_samples, dtype=np.float64)
    vel_int_y = np.zeros(n_samples, dtype=np.float64)
    vel_int_z = np.zeros(n_samples, dtype=np.float64)
    z_i = np.zeros(n_samples, dtype=np.float64)
    costs = np.zeros(n_samples, dtype=np.float64)

    predictions = None
    if return_predictions:
        predictions = np.zeros((n_samples, HORIZON + 1, 3), dtype=np.float64)
        predictions[:, 0, 0] = x
        predictions[:, 0, 1] = y
        predictions[:, 0, 2] = z

    for h in range(HORIZON):
        x_ref = inputs[:, h, 0]
        y_ref = inputs[:, h, 1]
        z_ref = inputs[:, h, 2]
        yaw_ref = inputs[:, h, 3]

        vel_sp_x = params.mpc_xy_p * (x_ref - x)
        vel_sp_y = params.mpc_xy_p * (y_ref - y)
        vel_sp_z = params.mpc_z_p * (z_ref - z)
        vel_xy_norm = np.sqrt(vel_sp_x * vel_sp_x + vel_sp_y * vel_sp_y)
        over_xy = (vel_xy_norm > MPC_XY_VEL_MAX) & (vel_xy_norm > 1e-8)
        vel_sp_x[over_xy] = vel_sp_x[over_xy] / vel_xy_norm[over_xy] * MPC_XY_VEL_MAX
        vel_sp_y[over_xy] = vel_sp_y[over_xy] / vel_xy_norm[over_xy] * MPC_XY_VEL_MAX
        vel_sp_z = np.clip(vel_sp_z, -MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DOWN)

        vel_dot_x = (vx - prev_vx) / CONTROL_PERIOD
        vel_dot_y = (vy - prev_vy) / CONTROL_PERIOD
        vel_dot_z = (vz - prev_vz) / CONTROL_PERIOD
        if MPC_VELD_LP > 1e-6:
            vel_dot_alpha = CONTROL_PERIOD / (CONTROL_PERIOD + 1.0 / (2.0 * np.pi * MPC_VELD_LP))
        else:
            vel_dot_alpha = 1.0
        vel_dot_lpf_x = prev_acc_x + vel_dot_alpha * (vel_dot_x - prev_acc_x)
        vel_dot_lpf_y = prev_acc_y + vel_dot_alpha * (vel_dot_y - prev_acc_y)
        vel_dot_lpf_z = prev_acc_z + vel_dot_alpha * (vel_dot_z - prev_acc_z)

        vel_error_x = vel_sp_x - vx
        vel_error_y = vel_sp_y - vy
        vel_error_z = vel_sp_z - vz
        acc_x = params.mpc_xy_vel_p_acc * vel_error_x + vel_int_x - params.mpc_xy_vel_d_acc * vel_dot_lpf_x
        acc_y = params.mpc_xy_vel_p_acc * vel_error_y + vel_int_y - params.mpc_xy_vel_d_acc * vel_dot_lpf_y
        acc_z = params.mpc_z_vel_p_acc * vel_error_z + vel_int_z - params.mpc_z_vel_d_acc * vel_dot_lpf_z

        body_z = np.column_stack([-acc_x, -acc_y, np.full(n_samples, A_OF_GRAVITY) - acc_z])
        body_z = normalize_batch(body_z)
        dot_z = np.clip(body_z[:, 2], -1.0, 1.0)
        tilt_angle = np.arccos(dot_z)
        tilt_over = tilt_angle > MPC_TILT_MAX
        if np.any(tilt_over):
            rejection = body_z[tilt_over].copy()
            rejection[:, 2] = 0.0
            rejection_norm = np.linalg.norm(rejection[:, :2], axis=1)
            tiny = rejection_norm < 1e-8
            rejection[tiny] = np.array([1.0, 0.0, 0.0])
            rejection = normalize_batch(rejection)
            body_z[tilt_over, 0] = np.sin(MPC_TILT_MAX) * rejection[:, 0]
            body_z[tilt_over, 1] = np.sin(MPC_TILT_MAX) * rejection[:, 1]
            body_z[tilt_over, 2] = np.cos(MPC_TILT_MAX)

        thrust_ned_z = acc_z * (MPC_THR_HOVER / A_OF_GRAVITY) - MPC_THR_HOVER
        cos_ned_body = body_z[:, 2].copy()
        cos_ned_body[np.abs(cos_ned_body) < 1e-6] = 1e-6
        collective_thrust = np.minimum(thrust_ned_z / cos_ned_body, -MPC_THR_MIN)
        thrust_sp = body_z * collective_thrust[:, None]

        thrust_sp_xy_norm = np.linalg.norm(thrust_sp[:, :2], axis=1)
        thrust_max_squared = MPC_THR_MAX * MPC_THR_MAX
        allocated_horizontal_thrust = np.minimum(thrust_sp_xy_norm, MPC_THR_XY_MARGIN)
        thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust * allocated_horizontal_thrust
        thrust_sp[:, 2] = np.maximum(thrust_sp[:, 2], -np.sqrt(np.maximum(0.0, thrust_z_max_squared)))

        thrust_max_xy_squared = thrust_max_squared - thrust_sp[:, 2] * thrust_sp[:, 2]
        thrust_max_xy = np.sqrt(np.maximum(0.0, thrust_max_xy_squared))
        over_thrust_xy = (thrust_sp_xy_norm > thrust_max_xy) & (thrust_sp_xy_norm > 1e-8)
        thrust_sp[over_thrust_xy, 0] = thrust_sp[over_thrust_xy, 0] / thrust_sp_xy_norm[over_thrust_xy] * thrust_max_xy[over_thrust_xy]
        thrust_sp[over_thrust_xy, 1] = thrust_sp[over_thrust_xy, 1] / thrust_sp_xy_norm[over_thrust_xy] * thrust_max_xy[over_thrust_xy]

        att_sp = thrust_to_attitude_simulator_batch(thrust_sp, yaw_ref)
        qe0 = np.sum(q * att_sp, axis=1)
        sgn = np.where(qe0 >= 0.0, 1.0, -1.0)
        omega_sp_x = 2.0 * MC_ROLL_P * sgn * (q[:, 0] * att_sp[:, 1] - q[:, 1] * att_sp[:, 0] - q[:, 2] * att_sp[:, 3] + q[:, 3] * att_sp[:, 2])
        omega_sp_y = 2.0 * MC_PITCH_P * sgn * (q[:, 0] * att_sp[:, 2] + q[:, 1] * att_sp[:, 3] - q[:, 2] * att_sp[:, 0] - q[:, 3] * att_sp[:, 1])
        omega_sp_z = 2.0 * MC_YAW_P * sgn * (q[:, 0] * att_sp[:, 3] - q[:, 1] * att_sp[:, 2] + q[:, 2] * att_sp[:, 1] - q[:, 3] * att_sp[:, 0])

        x += MODEL_POS_ALPHA[0] * vx * CONTROL_PERIOD
        y += MODEL_POS_ALPHA[1] * vy * CONTROL_PERIOD
        z += MODEL_POS_ALPHA[2] * vz * CONTROL_PERIOD
        z_i += z * CONTROL_PERIOD

        v_prev = np.column_stack([vx, vy, vz])
        vel_next = MODEL_VEL_A * v_prev + MODEL_VEL_B * np.column_stack([acc_x, acc_y, acc_z])
        vx, vy, vz = vel_next[:, 0], vel_next[:, 1], vel_next[:, 2]

        w_prev = np.column_stack([wx, wy, wz])
        q_dot = 0.5 * quat_mul_batch(q, np.column_stack([np.zeros(n_samples), wx, wy, wz]))
        q = normalize_batch(q + MODEL_ATT_BETA * q_dot * CONTROL_PERIOD)

        w_next = MODEL_W_C * w_prev + MODEL_W_D * np.column_stack([omega_sp_x, omega_sp_y, omega_sp_z])
        wx, wy, wz = w_next[:, 0], w_next[:, 1], w_next[:, 2]

        costs += (
            COST_Q["x"] * (x - target["x"]) ** 2
            + COST_Q["y"] * (y - target["y"]) ** 2
            + COST_Q["z"] * (z - target["z"]) ** 2
            + COST_Q["vx"] * (vx - target["vx"]) ** 2
            + COST_Q["vy"] * (vy - target["vy"]) ** 2
            + COST_Q["vz"] * (vz - target["vz"]) ** 2
            + COST_Q["zi"] * z_i ** 2
            + COST_R["x"] * (x_ref - target["x"]) ** 2
            + COST_R["y"] * (y_ref - target["y"]) ** 2
            + COST_R["z"] * (z_ref - target["z"]) ** 2
            + COST_R["yaw"] * yaw_ref ** 2
        )

        prev_vx = vx.copy()
        prev_vy = vy.copy()
        prev_vz = vz.copy()
        prev_acc_x = vel_dot_lpf_x.copy()
        prev_acc_y = vel_dot_lpf_y.copy()
        prev_acc_z = vel_dot_lpf_z.copy()
        vel_error_for_int_x = vel_error_x.copy()
        vel_error_for_int_y = vel_error_y.copy()
        vel_error_for_int_z = vel_error_z.copy()
        z_saturated = (
            ((thrust_sp[:, 2] >= -MPC_THR_MIN) & (vel_error_for_int_z >= 0.0))
            | ((thrust_sp[:, 2] <= -MPC_THR_MAX) & (vel_error_for_int_z <= 0.0))
        )
        vel_error_for_int_z[z_saturated] = 0.0
        vel_int_x += vel_error_for_int_x * params.mpc_xy_vel_i_acc * CONTROL_PERIOD
        vel_int_y += vel_error_for_int_y * params.mpc_xy_vel_i_acc * CONTROL_PERIOD
        vel_int_z += vel_error_for_int_z * params.mpc_z_vel_i_acc * CONTROL_PERIOD
        vel_int_z = np.clip(vel_int_z, -A_OF_GRAVITY, A_OF_GRAVITY)

        if predictions is not None:
            predictions[:, h + 1, 0] = x
            predictions[:, h + 1, 1] = y
            predictions[:, h + 1, 2] = z

    return predictions, costs


def simulate_single(
    initial_state: np.ndarray,
    input_seq: np.ndarray,
    target: dict,
    params: Params,
) -> np.ndarray:
    pred, _ = simulate_batch(initial_state, input_seq[None, :, :], target, params, True)
    return pred[0]


def optimize(
    initial_state: np.ndarray,
    target: dict,
    average_input: np.ndarray,
    params: Params,
    seed: int,
    n_samples: int,
    same_cycle_iterations: int,
    capture_debug: bool = False,
) -> tuple[np.ndarray, float, DebugResult | None]:
    best_input = average_input.copy()
    debug = None
    min_cost = np.inf

    for iteration in range(same_cycle_iterations):
        rng = np.random.default_rng(seed + iteration)
        iteration_sigma = SIGMA * (SIGMA_DECAY ** iteration)
        samples = generate_samples(best_input, n_samples, rng, iteration_sigma)
        _, costs = simulate_batch(initial_state, samples, target, params, False)
        elite_indices = np.argsort(costs)[:N_OF_ELITE]
        elite_samples = samples[elite_indices]
        elite_costs = costs[elite_indices]
        best_input = weighted_average(elite_samples, elite_costs)
        min_cost = float(elite_costs[0])

        if capture_debug and iteration == same_cycle_iterations - 1:
            elite_predictions, _ = simulate_batch(initial_state, elite_samples, target, params, True)
            best_prediction = simulate_single(initial_state, best_input, target, params)
            debug = DebugResult(
                samples=samples,
                costs=costs,
                elite_indices=elite_indices,
                elite_samples=elite_samples,
                elite_predictions=elite_predictions,
                best_input=best_input,
                best_prediction=best_prediction,
            )

    return best_input, min_cost, debug


def apply_first_input(state: np.ndarray, input_seq: np.ndarray, target: dict, params: Params) -> np.ndarray:
    next_state = state.copy()
    q = state[IDX["e0"]:IDX["e3"] + 1].astype(np.float64)
    q_norm = np.linalg.norm(q)
    q = q / q_norm if q_norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0])
    w_prev = state[IDX["wx"]:IDX["wz"] + 1].astype(np.float64)
    pos = state[IDX["x"]:IDX["z"] + 1].astype(np.float64)
    vel_prev = state[IDX["vx"]:IDX["vz"] + 1].astype(np.float64)

    x_ref, y_ref, z_ref, yaw_ref = input_seq[0]
    vel_sp = np.array([
        params.mpc_xy_p * (x_ref - pos[0]),
        params.mpc_xy_p * (y_ref - pos[1]),
        params.mpc_z_p * (z_ref - pos[2]),
    ], dtype=np.float64)
    vel_xy_norm = np.linalg.norm(vel_sp[:2])
    if vel_xy_norm > MPC_XY_VEL_MAX and vel_xy_norm > 1e-8:
        vel_sp[:2] = vel_sp[:2] / vel_xy_norm * MPC_XY_VEL_MAX
    vel_sp[2] = np.clip(vel_sp[2], -MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DOWN)

    vel_error = vel_sp - vel_prev
    acc_sp = np.array([
        params.mpc_xy_vel_p_acc * vel_error[0],
        params.mpc_xy_vel_p_acc * vel_error[1],
        params.mpc_z_vel_p_acc * vel_error[2],
    ], dtype=np.float64)

    body_z = np.array([-acc_sp[0], -acc_sp[1], A_OF_GRAVITY - acc_sp[2]], dtype=np.float64)
    body_z_norm = np.linalg.norm(body_z)
    body_z = body_z / body_z_norm if body_z_norm > 1e-8 else np.array([0.0, 0.0, 1.0])
    dot_z = np.clip(body_z[2], -1.0, 1.0)
    if np.arccos(dot_z) > MPC_TILT_MAX:
        rejection = body_z.copy()
        rejection[2] = 0.0
        rejection_norm = np.linalg.norm(rejection)
        if rejection_norm < 1e-8:
            rejection = np.array([1.0, 0.0, 0.0])
        else:
            rejection /= rejection_norm
        body_z = np.array([
            np.sin(MPC_TILT_MAX) * rejection[0],
            np.sin(MPC_TILT_MAX) * rejection[1],
            np.cos(MPC_TILT_MAX),
        ])

    thrust_ned_z = acc_sp[2] * (MPC_THR_HOVER / A_OF_GRAVITY) - MPC_THR_HOVER
    cos_ned_body = body_z[2] if abs(body_z[2]) >= 1e-6 else 1e-6
    collective_thrust = min(thrust_ned_z / cos_ned_body, -MPC_THR_MIN)
    thrust_sp = body_z * collective_thrust

    thrust_sp_xy_norm = np.linalg.norm(thrust_sp[:2])
    allocated_horizontal_thrust = min(thrust_sp_xy_norm, MPC_THR_XY_MARGIN)
    thrust_z_max_squared = MPC_THR_MAX * MPC_THR_MAX - allocated_horizontal_thrust * allocated_horizontal_thrust
    thrust_sp[2] = max(thrust_sp[2], -np.sqrt(max(0.0, thrust_z_max_squared)))
    thrust_max_xy = np.sqrt(max(0.0, MPC_THR_MAX * MPC_THR_MAX - thrust_sp[2] * thrust_sp[2]))
    if thrust_sp_xy_norm > thrust_max_xy and thrust_sp_xy_norm > 1e-8:
        thrust_sp[:2] = thrust_sp[:2] / thrust_sp_xy_norm * thrust_max_xy

    att_sp = thrust_to_attitude_simulator_batch(thrust_sp[None, :], np.array([yaw_ref]))[0]
    sgn = 1.0 if np.dot(q, att_sp) >= 0.0 else -1.0
    omega_sp = np.array([
        2.0 * MC_ROLL_P * sgn * (q[0] * att_sp[1] - q[1] * att_sp[0] - q[2] * att_sp[3] + q[3] * att_sp[2]),
        2.0 * MC_PITCH_P * sgn * (q[0] * att_sp[2] + q[1] * att_sp[3] - q[2] * att_sp[0] - q[3] * att_sp[1]),
        2.0 * MC_YAW_P * sgn * (q[0] * att_sp[3] - q[1] * att_sp[2] + q[2] * att_sp[1] - q[3] * att_sp[0]),
    ])

    pos_next = pos + MODEL_POS_ALPHA * vel_prev * CONTROL_PERIOD
    vel_next = MODEL_VEL_A * vel_prev + MODEL_VEL_B * acc_sp
    q_dot = 0.5 * quat_mul_batch(q[None, :], np.array([[0.0, w_prev[0], w_prev[1], w_prev[2]]]))[0]
    q_next = normalize_batch((q + MODEL_ATT_BETA * q_dot * CONTROL_PERIOD)[None, :])[0]
    w_next = MODEL_W_C * w_prev + MODEL_W_D * omega_sp

    next_state[IDX["e0"]:IDX["e3"] + 1] = q_next
    next_state[IDX["wx"]:IDX["wz"] + 1] = w_next
    next_state[IDX["x"]:IDX["z"] + 1] = pos_next
    next_state[IDX["vx"]:IDX["vz"] + 1] = vel_next
    return next_state


def run_rollout(
    initial_state: np.ndarray,
    target: dict,
    params: Params,
    duration: float,
    seed: int,
    n_samples: int,
    same_cycle_iterations: int,
) -> list[RolloutStep]:
    n_steps = max(1, int(np.ceil(duration / CONTROL_PERIOD))) + 1
    state = initial_state.copy()
    previous_best_input = make_initial_average_input(initial_state)
    steps = []

    for step in range(n_steps):
        average_input = shift_input(previous_best_input)
        best_input, cost, _ = optimize(
            state,
            target,
            average_input,
            params,
            seed + step * 1000,
            n_samples,
            same_cycle_iterations,
            capture_debug=False,
        )
        steps.append(
            RolloutStep(
                t=step * CONTROL_PERIOD,
                state=state.copy(),
                average_input=average_input.copy(),
                best_input=best_input.copy(),
                cost=cost,
            )
        )
        state = apply_first_input(state, best_input, target, params)
        previous_best_input = best_input

    return steps


def draw_debug(
    plots: dict,
    rollout: list[RolloutStep],
    target: dict,
    params: Params,
    step_idx: int,
    seed: int,
    n_samples: int,
    same_cycle_iterations: int,
):
    step = rollout[step_idx]
    _, _, debug = optimize(
        step.state,
        target,
        step.average_input,
        params,
        seed + step_idx * 1000,
        n_samples,
        same_cycle_iterations,
        capture_debug=True,
    )
    assert debug is not None

    axes = plots["axes"]
    if plots["colorbar"] is not None:
        plots["colorbar"].remove()
        plots["colorbar"] = None
    if plots["colorbar_xz"] is not None:
        plots["colorbar_xz"].remove()
        plots["colorbar_xz"] = None
    for ax in axes.values():
        ax.clear()
        ax.grid(True, alpha=0.3)

    all_first = debug.samples[:, 0, :]
    elite_first = debug.elite_samples[:, 0, :]
    current_xyz = step.state[IDX["x"]:IDX["z"] + 1]
    log_cost = np.log10(np.maximum(debug.costs, 1e-12))

    max_plot_samples = min(6000, len(all_first))
    sample_idx = np.linspace(0, len(all_first) - 1, max_plot_samples, dtype=int)
    horizon_idx = np.arange(HORIZON)
    pred_t = np.arange(HORIZON + 1) * CONTROL_PERIOD

    ax_xy = axes["xy"]
    xy_scatter = ax_xy.scatter(
        all_first[sample_idx, 0],
        all_first[sample_idx, 1],
        s=7,
        c=log_cost[sample_idx],
        cmap="viridis_r",
        alpha=0.65,
        linewidths=0,
    )
    plots["colorbar"] = plots["figures"]["xy"].colorbar(xy_scatter, ax=ax_xy)
    plots["colorbar"].set_label("log10(cost)")
    ax_xy.scatter(elite_first[:, 0], elite_first[:, 1], s=48, facecolors="none", edgecolors="red", linewidths=1.3, label="elite")
    ax_xy.scatter([target["x"]], [target["y"]], s=70, marker="x", color="black", linewidths=1.8, label="target")
    ax_xy.scatter([current_xyz[0]], [current_xyz[1]], s=80, marker="+", color="blue", linewidths=1.8, label="current")
    ax_xy.set_title(f"u0 x-y samples  t={step.t:.2f}s")
    ax_xy.set_xlabel("u0_x")
    ax_xy.set_ylabel("u0_y")
    ax_xy.axis("equal")
    ax_xy.legend(loc="best", fontsize=9)

    sample_predictions, _ = simulate_batch(step.state, debug.samples[sample_idx], target, params, True)

    for axis_i, axis_name in enumerate(("x", "y", "z")):
        ax_cost = axes[f"{axis_name}cost"]
        ax_cost.scatter(all_first[sample_idx, axis_i], debug.costs[sample_idx], s=7, color="#1f77b4", alpha=0.45, linewidths=0)
        ax_cost.scatter(elite_first[:, axis_i], debug.costs[debug.elite_indices], s=48, facecolors="none", edgecolors="red", linewidths=1.3, label="elite")
        ax_cost.axvline(target[axis_name], color="black", linestyle="--", linewidth=1.5, label=f"target_{axis_name}")
        ax_cost.set_title(f"u0_{axis_name} vs cost")
        ax_cost.set_xlabel(f"u0_{axis_name}")
        ax_cost.set_ylabel("cost")
        ax_cost.legend(loc="best", fontsize=9)

        ax_pred = axes[f"pred_{axis_name}"]
        for pred in sample_predictions[:200]:
            ax_pred.plot(pred_t, pred[:, axis_i], color="0.75", alpha=0.12, linewidth=1.0)
        for pred in debug.elite_predictions:
            ax_pred.plot(pred_t, pred[:, axis_i], color="red", alpha=0.68, linewidth=1.2)
        ax_pred.plot(pred_t, debug.best_prediction[:, axis_i], color="black", linewidth=2.0, label="weighted elite")
        ax_pred.axhline(target[axis_name], color="black", linestyle="--", linewidth=1.5, label=f"target {axis_name}")
        ax_pred.set_title(f"predicted {axis_name} trajectories")
        ax_pred.set_xlabel("prediction time [s]")
        ax_pred.set_ylabel(axis_name)
        ax_pred.legend(loc="best", fontsize=9)

        ax_seq = axes[f"{axis_name}seq"]
        for sample in debug.samples[sample_idx[:250]]:
            ax_seq.plot(horizon_idx, sample[:, axis_i], color="0.75", alpha=0.10, linewidth=1.0)
        for elite in debug.elite_samples:
            ax_seq.plot(horizon_idx, elite[:, axis_i], color="red", alpha=0.55, linewidth=1.2)
        ax_seq.plot(horizon_idx, step.average_input[:, axis_i], color="blue", linewidth=2.0, label="average input")
        ax_seq.plot(horizon_idx, debug.best_input[:, axis_i], color="black", linewidth=2.0, label="weighted elite")
        ax_seq.axhline(target[axis_name], color="black", linestyle="--", linewidth=1.0, alpha=0.75)
        ax_seq.set_title(f"{axis_name} setpoint sequences")
        ax_seq.set_xlabel("horizon index")
        ax_seq.set_ylabel(f"u_{axis_name}")
        ax_seq.legend(loc="best", fontsize=9)

    ax_xz = axes["xz"]
    xz_scatter = ax_xz.scatter(
        all_first[sample_idx, 0],
        all_first[sample_idx, 2],
        s=7,
        c=log_cost[sample_idx],
        cmap="viridis_r",
        alpha=0.65,
        linewidths=0,
    )
    plots["colorbar_xz"] = plots["figures"]["xz"].colorbar(xz_scatter, ax=ax_xz)
    plots["colorbar_xz"].set_label("log10(cost)")
    ax_xz.scatter(elite_first[:, 0], elite_first[:, 2], s=48, facecolors="none", edgecolors="red", linewidths=1.3, label="elite")
    ax_xz.scatter([target["x"]], [target["z"]], s=70, marker="x", color="black", linewidths=1.8, label="target")
    ax_xz.scatter([current_xyz[0]], [current_xyz[2]], s=80, marker="+", color="blue", linewidths=1.8, label="current")
    ax_xz.set_title("u0 x-z samples")
    ax_xz.set_xlabel("u0_x")
    ax_xz.set_ylabel("u0_z")
    ax_xz.axis("equal")
    ax_xz.legend(loc="best", fontsize=9)

    ax_hist = axes["hist"]
    ax_hist.hist(all_first[:, 0], bins=80, alpha=0.45, color="#1f77b4", label="u0_x")
    ax_hist.hist(all_first[:, 1], bins=80, alpha=0.45, color="#ff7f0e", label="u0_y")
    ax_hist.hist(all_first[:, 2], bins=80, alpha=0.45, color="#2ca02c", label="u0_z")
    ax_hist.axvline(np.mean(elite_first[:, 0]), color="#1f77b4", linestyle="--", linewidth=1.7)
    ax_hist.axvline(np.mean(elite_first[:, 1]), color="#ff7f0e", linestyle="--", linewidth=1.7)
    ax_hist.axvline(np.mean(elite_first[:, 2]), color="#2ca02c", linestyle="--", linewidth=1.7)
    ax_hist.set_title("u0_x / u0_y / u0_z distribution")
    ax_hist.set_xlabel("u0 value")
    ax_hist.set_ylabel("count")
    ax_hist.legend(loc="best", fontsize=9)

    draw_input_error_summary(axes["input_error"], rollout)

    for fig in plots["figures"].values():
        fig.canvas.draw_idle()


def calc_future_input_error_stats(rollout: list[RolloutStep]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_steps = len(rollout)
    max_horizon = min(HORIZON - 1, n_steps - 1)
    horizon = np.arange(1, max_horizon + 1)
    rmse = np.full((max_horizon, 4), np.nan, dtype=np.float64)
    mae = np.full((max_horizon, 4), np.nan, dtype=np.float64)
    bias = np.full((max_horizon, 4), np.nan, dtype=np.float64)

    actual_u0 = np.array([step.best_input[0] for step in rollout], dtype=np.float64)
    for h in horizon:
        predicted = np.array([step.best_input[h] for step in rollout[:-h]], dtype=np.float64)
        actual = actual_u0[h:]
        error = predicted - actual
        rmse[h - 1] = np.sqrt(np.mean(error * error, axis=0))
        mae[h - 1] = np.mean(np.abs(error), axis=0)
        bias[h - 1] = np.mean(error, axis=0)

    return horizon, rmse, mae, bias


def draw_input_error_summary(ax, rollout: list[RolloutStep]) -> None:
    ax.clear()
    ax.grid(True, alpha=0.3)
    horizon, rmse, mae, bias = calc_future_input_error_stats(rollout)
    if len(horizon) == 0:
        ax.set_title("future optimal input error")
        ax.text(0.5, 0.5, "not enough rollout steps", transform=ax.transAxes, ha="center", va="center")
        return

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    labels = ["x", "y", "z", "yaw"]
    time_ahead = horizon * CONTROL_PERIOD
    for axis_i, label in enumerate(labels):
        ax.plot(time_ahead, rmse[:, axis_i], color=colors[axis_i], linewidth=2.0, label=f"{label} rmse")
        ax.plot(time_ahead, bias[:, axis_i], color=colors[axis_i], linewidth=1.1, linestyle="--", alpha=0.75, label=f"{label} bias")

    one_step = ", ".join(f"{labels[i]}={rmse[0, i]:.3g}" for i in range(4))
    overall = ", ".join(f"{labels[i]}={np.nanmean(rmse[:, i]):.3g}" for i in range(4))
    ax.text(
        0.01,
        0.99,
        f"1-step RMSE: {one_step}\nmean RMSE: {overall}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
    )
    ax.set_title("future predicted input vs actual applied input")
    ax.set_xlabel("prediction lead time [s]")
    ax.set_ylabel("predicted best_input[h] - actual best_input[0]")
    ax.legend(loc="best", fontsize=8, ncol=2)


def make_plots() -> dict:
    figure_specs = [
        ("xy", "u0 x-y samples"),
        ("xcost", "u0_x vs cost"),
        ("ycost", "u0_y vs cost"),
        ("zcost", "u0_z vs cost"),
        ("pred_x", "predicted x trajectories"),
        ("pred_y", "predicted y trajectories"),
        ("pred_z", "predicted z trajectories"),
        ("xseq", "x setpoint sequences"),
        ("yseq", "y setpoint sequences"),
        ("zseq", "z setpoint sequences"),
        ("xz", "u0 x-z samples"),
        ("hist", "u0 distribution"),
        ("input_error", "future input error"),
    ]
    figures = {}
    axes = {}
    for key, title in figure_specs:
        fig, ax = plt.subplots(figsize=(7.0, 5.2), num=title)
        figures[key] = fig
        axes[key] = ax
    return {"figures": figures, "axes": axes, "colorbar": None, "colorbar_xz": None}


def main():
    parser = argparse.ArgumentParser(
        description="Debug reproduction of the current MCMPC sampling simulator."
    )
    parser.add_argument("--target", type=parse_vec4, default=parse_vec4("0.0,0.0,-0.8,0.0"), help="target x,y,z,yaw")
    parser.add_argument("--current", type=parse_vec3, default=parse_vec3("0.0,0.0,-0.8"), help="initial/current x,y,z at t=0")
    parser.add_argument("--velocity", type=parse_vec3, default=parse_vec3("0.0,0.0,0.0"), help="initial vx,vy,vz")
    parser.add_argument("--yaw", type=float, default=0.0, help="initial yaw")
    parser.add_argument("--time", type=float, default=0.0, help="initial display time [s]")
    parser.add_argument("--duration", type=float, default=5.0, help="rollout duration from 0s [s]")
    parser.add_argument("--samples", type=int, default=N_OF_SAMPLES, help="number of input samples")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--same-cycle-iterations", type=int, default=SAME_CYCLE_ITERATIONS, help="repeat MPC optimization within one control cycle")
    parser.add_argument("--profile", choices=("px4", "simulation", "current", "legacy"), default="simulation", help="parameter profile matching mcmpc_viewer")
    parser.add_argument("--simulation-params", action="store_true", help="deprecated alias for --profile simulation")
    args = parser.parse_args()
    if args.samples < N_OF_ELITE:
        parser.error(f"--samples must be >= {N_OF_ELITE}")
    if args.same_cycle_iterations < 1:
        parser.error("--same-cycle-iterations must be >= 1")

    profile = "simulation" if args.simulation_params else args.profile
    params = Params(mpc_xy_p=apply_param_profile(profile))
    target = make_target(args.target)
    initial_state = make_state(args.current, args.velocity, args.yaw)
    rollout = run_rollout(
        initial_state,
        target,
        params,
        args.duration,
        args.seed,
        args.samples,
        args.same_cycle_iterations,
    )

    max_idx = len(rollout) - 1
    initial_idx = int(np.clip(round(args.time / CONTROL_PERIOD), 0, max_idx))
    plots = make_plots()
    draw_debug(plots, rollout, target, params, initial_idx, args.seed, args.samples, args.same_cycle_iterations)

    control_fig = plt.figure("time control", figsize=(7.0, 1.3))
    slider_ax = control_fig.add_axes([0.14, 0.35, 0.78, 0.28])
    time_slider = Slider(
        slider_ax,
        "time [s]",
        0.0,
        rollout[-1].t,
        valinit=rollout[initial_idx].t,
        valstep=CONTROL_PERIOD,
    )

    def on_time_change(value):
        step_idx = int(np.clip(round(value / CONTROL_PERIOD), 0, max_idx))
        draw_debug(plots, rollout, target, params, step_idx, args.seed, args.samples, args.same_cycle_iterations)
        control_fig.canvas.draw_idle()

    time_slider.on_changed(on_time_change)
    plt.show()


if __name__ == "__main__":
    main()
