import argparse
import os
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

DEFAULT_CSV = "/home/ros2/ws_mcmpc/src/visualize_mcmpc/csv/offboard_control_log_sinx_8.csv"
# DEFAULT_CSV = "/home/ros2/ws_mcmpc/src/visualize_mcmpc/csv/offboard_control_log_sin_real3.csv"
# DEFAULT_CSV = "/home/ros2/ws_mcmpc/src/visualize_mcmpc/csv/offboard_control_log_step7.csv"
# DEFAULT_CSV = "/home/ros2/ws_mcmpc/src/visualize_mcmpc/csv/offboard_control_log_sin2.csv"

DT = 0.02
SIMULATION = True
A_OF_GRAVITY = 9.80665
ANGULAR_ACCEL_LP = 30.0
CA_MINIMUM_YAW_MARGIN = 0.15
MPC_ACC_DECOUPLE = False
RATE_DELAY_STEPS = 1
MOTOR_COMMAND_DELAY_STEPS = 0
ACTUATOR_LOG_SHIFT_STEPS = -1
TARGET_LOCAL_SP_DELAY_STEPS = 2
RATE_INT_SYNC_PERIOD = 1.5
PREDICTION_HORIZON = 75
PREDICTION_INTERVAL = 1.5
USE_RATE_INT_BIAS_TORQUE = True
USE_RATE_INT_BIAS_TORQUE_WITH_ACTUATOR_LOG = True
USE_ACCELERATION_BIAS_OBSERVER = True
USE_LIVOX_IMU_ACCELERATION = True
USE_VELOCITY_DELTA_BIAS_OBSERVER = True
USE_ACCELERATION_BIAS_TREND_PREDICTION = True
USE_Z_POSITION_VELOCITY_BIAS_CORRECTION = True
ACCELERATION_BIAS_ALPHA = 0.05
VELOCITY_DELTA_BIAS_ALPHA = 0.10
Z_POSITION_VELOCITY_BIAS_ALPHA = 0.05
Z_POSITION_VELOCITY_BIAS_LIMIT = 0.5
ACCELERATION_BIAS_TREND_LOOKBACK_STEPS = 10
ACCELERATION_BIAS_TREND_RATE_LIMIT = 0.30
ACCELERATION_BIAS_LIMIT = np.array([3.0, 3.0, 2.0], dtype=float)
ACCELERATION_BIAS_DEADBAND = np.array([0.02, 0.02, 0.01], dtype=float)
VELOCITY_DELTA_BIAS_AXES = np.array([True, True, True], dtype=bool)
TRANSLATION_BIAS_AXES = np.array([True, True, True], dtype=bool)

if SIMULATION:
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
    MPC_TKO_RAMP_T = 3.0
    MPC_THR_HOVER = 0.6295
    MPC_THR_MIN = 0.10
    MPC_THR_MAX = 0.90
    MPC_THR_XY_MARGIN = 0.30
    MPC_TILT_MAX = np.deg2rad(45.0)
    MPC_VELD_LP = 5.0

    MC_ROLL_P = 3.30
    MC_PITCH_P = 3.30
    MC_YAW_P = 2.80
    MC_YAW_WEIGHT = 0.50
    MC_ROLLRATE_MAX = np.deg2rad(220.0)
    MC_PITCHRATE_MAX = np.deg2rad(220.0)
    MC_YAWRATE_MAX = np.deg2rad(200.0)
    MC_ROLLRATE_P = 0.150
    MC_PITCHRATE_P = 0.150
    MC_YAWRATE_P = 0.500
    MC_ROLLRATE_K = 1.0
    MC_PITCHRATE_K = 1.0
    MC_YAWRATE_K = 1.0
    MC_ROLLRATE_D = 0.0035
    MC_PITCHRATE_D = 0.0035
    MC_YAWRATE_D = 0.00
    MC_ROLLRATE_I = 0.20
    MC_PITCHRATE_I = 0.20
    MC_YAWRATE_I = 0.100
    MC_ROLLRATE_FF = 0.0
    MC_PITCHRATE_FF = 0.0
    MC_YAWRATE_FF = 0.0
    MC_RR_INT_LIM = 0.30
    MC_PR_INT_LIM = 0.30
    MC_YR_INT_LIM = 0.30
    MC_YAW_TQ_CUTOFF = 2.0
    DRONE_MASS = 2.3
    DRONE_INERTIA = np.diag([0.0418, 0.04226, 0.05619])
else:
    MPC_XY_P = 0.95
    MPC_Z_P = 1.00
    MPC_XY_VEL_P_ACC = 2.00
    MPC_XY_VEL_I_ACC = 0.40
    MPC_XY_VEL_D_ACC = 0.20
    MPC_Z_VEL_P_ACC = 4.00
    MPC_Z_VEL_I_ACC = 2.00
    MPC_Z_VEL_D_ACC = 0.00
    MPC_XY_VEL_MAX = 43.2 / 3.6
    MPC_Z_VEL_MAX_UP = 10.8 / 3.6
    MPC_Z_VEL_MAX_DOWN = 5.4 / 3.6
    MPC_TKO_RAMP_T = 3.0
    MPC_THR_HOVER = -0.65
    MPC_THR_MIN = 0.12
    MPC_THR_MAX = 1.00
    MPC_THR_XY_MARGIN = 0.30
    MPC_TILT_MAX = np.deg2rad(45.0)
    MPC_VELD_LP = 5.0

    MC_ROLL_P = 4.00
    MC_PITCH_P = 4.00
    MC_YAW_P = 2.80
    MC_YAW_WEIGHT = 0.40
    MC_ROLLRATE_MAX = np.deg2rad(220.0)
    MC_PITCHRATE_MAX = np.deg2rad(220.0)
    MC_YAWRATE_MAX = np.deg2rad(200.0)
    MC_ROLLRATE_P = 0.150
    MC_PITCHRATE_P = 0.150
    MC_YAWRATE_P = 0.200
    MC_ROLLRATE_K = 1.0
    MC_PITCHRATE_K = 1.0
    MC_YAWRATE_K = 1.0
    MC_ROLLRATE_D = 0.0030
    MC_PITCHRATE_D = 0.0030
    MC_YAWRATE_D = 0.00
    MC_ROLLRATE_I = 0.20
    MC_PITCHRATE_I = 0.20
    MC_YAWRATE_I = 0.100
    MC_ROLLRATE_FF = 0.0
    MC_PITCHRATE_FF = 0.0
    MC_YAWRATE_FF = 0.0
    MC_RR_INT_LIM = 0.30
    MC_PR_INT_LIM = 0.30
    MC_YR_INT_LIM = 0.30
    MC_YAW_TQ_CUTOFF = 2.0
    DRONE_MASS = 1.62
    DRONE_INERTIA = np.diag([0.0418, 0.04226, 0.05619])

# FRD座標系 SDFはFLU
ROTOR_POSITIONS = np.array([
    [0.180655, 0.180655, 0.0],   # rotor_1 front right, ccw
    [-0.180655, -0.180655, 0.0], # rotor_2 back left, ccw
    [0.180655, -0.180655, 0.0],  # rotor_3 front left, cw
    [-0.180655, 0.180655, 0.0],  # rotor_4 back right, cw
], dtype=float)
ROTOR_YAW_SIGNS = np.array([1.0, 1.0, -1.0, -1.0], dtype=float)
PX4_QUAD_X_MIX = np.array([
    [-0.70710678, 0.70710678, 1.0, -1.0],
    [0.70710678, -0.70710678, 1.0, -1.0],
    [0.70710678, 0.70710678, -1.0, -1.0],
    [-0.70710678, -0.70710678, -1.0, -1.0],
], dtype=float)
PX4_QUAD_X_MIX_INV = np.linalg.inv(PX4_QUAD_X_MIX)
PX4_ACTUATOR_MIN = np.zeros(4, dtype=float)
PX4_ACTUATOR_MAX = np.ones(4, dtype=float)
MOTOR_CONSTANT_SDF = 1.09e-5
MOTOR_THRUST_SCALE = 0.9429863114169338
MOTOR_CONSTANT = MOTOR_CONSTANT_SDF * MOTOR_THRUST_SCALE
MOMENT_CONSTANT = 5.0e-9
MOTOR_SPEED_MODEL = "bench_table"
MOTOR_INPUT_SCALING = 1122.0
MAX_ROT_VELOCITY = 1032.0
MOTOR_TIME_CONSTANT_UP = 0.004
MOTOR_TIME_CONSTANT_DOWN = 0.070
ROTOR_VELOCITY_SLOWDOWN_SIM = 10.0
BENCH_THROTTLE = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float) / 100.0
BENCH_RPM = np.array([4042, 4469, 4855, 5301, 5780, 6298, 6800, 7281, 7679, 8096, 8468, 8867, 9257, 9675, 9857], dtype=float)
BENCH_MIN_THROTTLE = BENCH_THROTTLE[0]
BENCH_MIN_RPM = BENCH_RPM[0]
BENCH_MAX_RPM = BENCH_RPM[-1]

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


def progress(message):
    print(f"[PROGRESS] {message}", flush=True)


def select_base_timestamp_column(df):
    for col in ("sync_timestamp", "local_pos_timestamp", "attitude_timestamp", "control_time_s"):
        if col in df.columns:
            return col
    return None


def timestamp_values_us(df, timestamp_col):
    values = df[timestamp_col].to_numpy(float)
    if timestamp_col == "control_time_s":
        return values * 1.0e6
    return values


def align_topic_columns_by_timestamp(df, timestamp_col, value_cols, base_timestamp_col=None):
    if base_timestamp_col is None:
        base_timestamp_col = select_base_timestamp_column(df)
    if timestamp_col not in df.columns or base_timestamp_col not in df.columns:
        return df

    value_cols = [col for col in value_cols if col in df.columns]
    if not value_cols:
        return df

    base_time = timestamp_values_us(df, base_timestamp_col)
    topic_time = timestamp_values_us(df, timestamp_col)
    valid_base = np.isfinite(base_time) & (base_time > 0.0)
    valid_topic = np.isfinite(topic_time) & (topic_time > 0.0)
    if not np.any(valid_base) or not np.any(valid_topic):
        return df

    topic = df.loc[valid_topic, [timestamp_col] + value_cols].copy()
    topic[timestamp_col] = topic[timestamp_col].astype(float)
    topic = topic.sort_values(timestamp_col).drop_duplicates(timestamp_col, keep="last")
    topic = topic.rename(columns={timestamp_col: "_topic_timestamp"})
    base = pd.DataFrame({
        "_row_index": np.arange(len(df), dtype=np.int64),
        "_base_timestamp": base_time,
    })
    aligned = pd.merge_asof(
        base.loc[valid_base].sort_values("_base_timestamp"),
        topic.sort_values("_topic_timestamp"),
        left_on="_base_timestamp",
        right_on="_topic_timestamp",
        direction="backward",
    ).sort_values("_row_index")

    rows = aligned["_row_index"].to_numpy(int)
    for col in value_cols:
        df.loc[rows, col] = aligned[col].to_numpy()
    df.loc[rows, timestamp_col] = aligned["_topic_timestamp"].to_numpy()

    matched = int(np.count_nonzero(np.isfinite(aligned["_topic_timestamp"].to_numpy(float))))
    print(
        f"[INFO] aligned {len(value_cols)} columns from {timestamp_col} "
        f"to {base_timestamp_col}: {matched}/{len(rows)} rows"
    )
    return df


def align_timestamped_topics(df):
    topic_groups = [
        ("local_sp_timestamp", [
            "local_sp_x", "local_sp_y", "local_sp_z", "local_sp_yaw",
            "local_sp_vx", "local_sp_vy", "local_sp_vz",
            "local_sp_ax", "local_sp_ay", "local_sp_az",
            "local_sp_yawspeed",
        ]),
        ("att_sp_timestamp", ["att_sp_qw", "att_sp_qx", "att_sp_qy", "att_sp_qz"]),
        ("rate_sp_timestamp", [
            "rate_sp_roll", "rate_sp_pitch", "rate_sp_yaw",
            "rate_sp_thrust_x", "rate_sp_thrust_y", "rate_sp_thrust_z",
        ]),
        ("thrust_sp_timestamp", ["thrust_sp_x", "thrust_sp_y", "thrust_sp_z"]),
        ("torque_sp_timestamp", ["torque_sp_x", "torque_sp_y", "torque_sp_z"]),
        ("hover_thrust_timestamp", ["hover_thrust", "hover_thrust_valid"]),
        ("takeoff_status_timestamp", ["takeoff_state", "takeoff_tilt_limit"]),
        ("land_detected_timestamp", ["landed", "ground_contact", "maybe_landed"]),
        ("actuator_timestamp", [f"actuator_motor_{i}" for i in range(12)]),
        ("rate_ctrl_status_timestamp", RATE_INT_COLUMNS),
    ]
    for timestamp_col, value_cols in topic_groups:
        df = align_topic_columns_by_timestamp(df, timestamp_col, value_cols)
    return df


def add_livox_imu_acceleration_columns(df):
    required = [
        "livox_imu_linear_acceleration_x",
        "livox_imu_linear_acceleration_y",
        "livox_imu_linear_acceleration_z",
        "livox_imu_orientation_x",
        "livox_imu_orientation_y",
        "livox_imu_orientation_z",
        "livox_imu_orientation_w",
    ]
    if not all(col in df.columns for col in required):
        return df

    acc_body = df[
        [
            "livox_imu_linear_acceleration_x",
            "livox_imu_linear_acceleration_y",
            "livox_imu_linear_acceleration_z",
        ]
    ].to_numpy(float)
    quat_xyzw = df[
        [
            "livox_imu_orientation_x",
            "livox_imu_orientation_y",
            "livox_imu_orientation_z",
            "livox_imu_orientation_w",
        ]
    ].to_numpy(float)
    acc_local = np.full((len(df), 3), np.nan, dtype=float)
    for i, (acc, q_xyzw) in enumerate(zip(acc_body, quat_xyzw)):
        if not np.all(np.isfinite(acc)) or not np.all(np.isfinite(q_xyzw)):
            continue
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
        acc_world = quat_to_rotmat(q_wxyz) @ acc - np.array([0.0, 0.0, A_OF_GRAVITY])
        # Livox IMU is in ROS world convention here: x/y match local horizontal axes,
        # z is opposite PX4 NED z.
        acc_local[i] = np.array([acc_world[0], acc_world[1], -acc_world[2]])

    df["livox_imu_acc_x"] = acc_local[:, 0]
    df["livox_imu_acc_y"] = acc_local[:, 1]
    df["livox_imu_acc_z"] = acc_local[:, 2]
    valid_count = int(np.count_nonzero(np.all(np.isfinite(acc_local), axis=1)))
    print(f"[INFO] computed livox_imu_acc_*: {valid_count}/{len(df)} rows")
    return df


def apply_synced_state_columns(df):
    if "sync_timestamp" in df.columns:
        sync_timestamp = df["sync_timestamp"].to_numpy(float)
        valid_sync_time = np.isfinite(sync_timestamp) & (sync_timestamp > 1.0e12)
        dropped = len(df) - int(np.count_nonzero(valid_sync_time))
        if dropped > 0:
            print(f"[INFO] dropped {dropped} rows with invalid sync_timestamp")
        df = df.loc[valid_sync_time].reset_index(drop=True)
        sync_timestamp = df["sync_timestamp"].to_numpy(float)
        if len(sync_timestamp) > 0:
            df["control_time_s"] = (sync_timestamp - float(sync_timestamp[0])) * 1.0e-6
            print("[INFO] using sync_timestamp as control_time_s")
    elif "sync_time_s" in df.columns:
        sync_time = df["sync_time_s"].to_numpy(float)
        valid_sync_time = np.isfinite(sync_time) & (sync_time > 1.0e6)
        dropped = len(df) - int(np.count_nonzero(valid_sync_time))
        if dropped > 0:
            print(f"[INFO] dropped {dropped} rows with invalid sync_time_s")
        df = df.loc[valid_sync_time].reset_index(drop=True)
        sync_time = df["sync_time_s"].to_numpy(float)
        if len(sync_time) > 0:
            df["control_time_s"] = sync_time - float(sync_time[0])
            print("[INFO] using sync_time_s as control_time_s")

    sync_mapping = {
        "sync_pos_x": "pos_x",
        "sync_pos_y": "pos_y",
        "sync_pos_z": "pos_z",
        "sync_vel_x": "vel_x",
        "sync_vel_y": "vel_y",
        "sync_vel_z": "vel_z",
        "sync_roll": "roll",
        "sync_pitch": "pitch",
        "sync_yaw": "yaw",
        "sync_angular_vel_x": "angular_vel_x",
        "sync_angular_vel_y": "angular_vel_y",
        "sync_angular_vel_z": "angular_vel_z",
    }
    available_sync_cols = [src for src in sync_mapping if src in df.columns]
    if available_sync_cols:
        for src, dst in sync_mapping.items():
            if src not in df.columns or dst not in df.columns:
                continue
            sync_value = df[src].to_numpy(float)
            sync_mask = np.isfinite(sync_value)
            df.loc[sync_mask, dst] = sync_value[sync_mask]
        print(f"[INFO] using {len(available_sync_cols)} sync_* columns for state")

    df = align_timestamped_topics(df)
    df = add_livox_imu_acceleration_columns(df)

    return df


def preprocess_log(df, step_dt):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df[np.isfinite(df["control_time_s"].to_numpy(float))].reset_index(drop=True)
    df = apply_synced_state_columns(df)

    t = df["control_time_s"].to_numpy(float)
    reset_indices = np.flatnonzero(np.diff(t) <= 0.0) + 1
    starts = np.r_[0, reset_indices]
    ends = np.r_[reset_indices, len(df)]
    segment_id = np.zeros(len(df), dtype=np.int64)
    for segment_idx, (start, end) in enumerate(zip(starts, ends)):
        segment_id[start:end] = segment_idx
    df["_segment_id"] = segment_id

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


def add_delayed_target_columns(df, delay_steps=TARGET_LOCAL_SP_DELAY_STEPS):
    if delay_steps <= 0:
        return df

    df = df.copy()
    target_cols = ["target_x", "target_y", "target_z", "target_yaw"]
    group_key = "_segment_id" if "_segment_id" in df.columns else None
    for col in target_cols:
        if col not in df.columns:
            continue
        delayed = (
            df.groupby(group_key, sort=False)[col].shift(delay_steps)
            if group_key is not None
            else df[col].shift(delay_steps)
        )
        df[f"{col}_delayed"] = delayed.fillna(df[col])
    return df


def add_actuator_input_columns(df, shift_steps=ACTUATOR_LOG_SHIFT_STEPS):
    actuator_cols = [f"actuator_motor_{motor_idx}" for motor_idx in range(4)]
    if not all(col in df.columns for col in actuator_cols):
        return df

    df = df.copy()
    shifted = df[actuator_cols]
    if shift_steps != 0:
        group_key = "_segment_id" if "_segment_id" in df.columns else None
        shifted = (
            df.groupby(group_key, sort=False)[actuator_cols].shift(shift_steps)
            if group_key is not None
            else df[actuator_cols].shift(shift_steps)
        )
        print(
            f"[INFO] shifted actuator input columns by {shift_steps} sample(s) "
            f"for actuator_log prediction"
        )
    for motor_idx, col in enumerate(actuator_cols):
        df[f"actuator_motor_input_{motor_idx}"] = shifted[col]
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


def quat_to_rotmat(q):
    q0, q1, q2, q3 = normalize(q)
    return np.array([
        [1.0 - 2.0 * (q2 * q2 + q3 * q3), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
        [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1 * q1 + q3 * q3), 2.0 * (q2 * q3 - q0 * q1)],
        [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2)],
    ])


def mix_px4_quad_x(thrust_sp, torque_sp):
    collective = -float(thrust_sp[2])
    roll, pitch, yaw = np.asarray(torque_sp, dtype=float)
    return np.array([
        collective - 0.70710678 * roll + 0.70710678 * pitch + yaw,
        collective + 0.70710678 * roll - 0.70710678 * pitch + yaw,
        collective + 0.70710678 * roll + 0.70710678 * pitch - yaw,
        collective - 0.70710678 * roll - 0.70710678 * pitch - yaw,
    ])


def allocate_px4_quad_x(control_sp):
    roll_mix = PX4_QUAD_X_MIX[:, 0]
    pitch_mix = PX4_QUAD_X_MIX[:, 1]
    yaw_mix = PX4_QUAD_X_MIX[:, 2]
    thrust_z_mix = PX4_QUAD_X_MIX[:, 3]

    motor_raw = (
        roll_mix * control_sp[0]
        + pitch_mix * control_sp[1]
        + thrust_z_mix * control_sp[3]
    )
    desaturation_steps = [
        (thrust_z_mix, True, PX4_ACTUATOR_MAX),
        (roll_mix, False, PX4_ACTUATOR_MAX),
        (pitch_mix, False, PX4_ACTUATOR_MAX),
    ]
    for desaturation_vector, increase_only, actuator_max in desaturation_steps:
        motor_raw = desaturate_motor_outputs(motor_raw, desaturation_vector, increase_only, actuator_max)

    motor_raw += yaw_mix * control_sp[2]
    yaw_actuator_max = PX4_ACTUATOR_MAX + (PX4_ACTUATOR_MAX - PX4_ACTUATOR_MIN) * CA_MINIMUM_YAW_MARGIN
    desaturation_steps = [
        (yaw_mix, False, yaw_actuator_max),
        (thrust_z_mix, True, PX4_ACTUATOR_MAX),
    ]
    for desaturation_vector, increase_only, actuator_max in desaturation_steps:
        motor_raw = desaturate_motor_outputs(motor_raw, desaturation_vector, increase_only, actuator_max)

    motor_setpoint = np.clip(motor_raw, PX4_ACTUATOR_MIN, PX4_ACTUATOR_MAX)
    allocated_control = PX4_QUAD_X_MIX_INV @ motor_setpoint
    unallocated_control = control_sp - allocated_control
    return motor_setpoint, unallocated_control


def desaturate_motor_outputs(motor_raw, desaturation_vector, increase_only, actuator_max):
    motor_raw = motor_raw.copy()
    k_min = 0.0
    k_max = 0.0
    for value, desat, minimum, maximum in zip(motor_raw, desaturation_vector, PX4_ACTUATOR_MIN, actuator_max):
        if abs(desat) < 0.2:
            continue
        if value < minimum:
            k = (minimum - value) / desat
            k_min = min(k_min, k)
            k_max = max(k_max, k)
        if value > maximum:
            k = (maximum - value) / desat
            k_min = min(k_min, k)
            k_max = max(k_max, k)
    gain = k_min + k_max
    if increase_only and gain < 0.0:
        return motor_raw

    motor_raw += gain * desaturation_vector
    k_min = 0.0
    k_max = 0.0
    for value, desat, minimum, maximum in zip(motor_raw, desaturation_vector, PX4_ACTUATOR_MIN, actuator_max):
        if abs(desat) < 0.2:
            continue
        if value < minimum:
            k = (minimum - value) / desat
            k_min = min(k_min, k)
            k_max = max(k_max, k)
        if value > maximum:
            k = (maximum - value) / desat
            k_min = min(k_min, k)
            k_max = max(k_max, k)
    motor_raw += 0.5 * (k_min + k_max) * desaturation_vector
    return motor_raw


def motor_speed_from_setpoint(motor_setpoint, step_dt, prev_motor_speed=None):
    actuator_for_speed = np.clip(motor_setpoint, 0.0, 1.0)
    if MOTOR_SPEED_MODEL == "bench_table":
        rpm_ref = np.interp(actuator_for_speed, BENCH_THROTTLE, BENCH_RPM)
        below_table = actuator_for_speed < BENCH_MIN_THROTTLE
        rpm_ref[below_table] = actuator_for_speed[below_table] / BENCH_MIN_THROTTLE * BENCH_MIN_RPM
        rpm_ref = np.clip(rpm_ref, 0.0, BENCH_MAX_RPM)
        motor_speed_ref = rpm_ref * (2.0 * np.pi / 60.0)
    else:
        motor_speed_ref = np.clip(actuator_for_speed * MOTOR_INPUT_SCALING, 0.0, MAX_ROT_VELOCITY)
    if prev_motor_speed is None:
        return motor_speed_ref

    max_motor_speed = max(MAX_ROT_VELOCITY, BENCH_MAX_RPM * (2.0 * np.pi / 60.0))
    motor_speed_prev = np.clip(np.asarray(prev_motor_speed, dtype=float)[:4], 0.0, max_motor_speed)
    motor_tau = np.where(motor_speed_ref > motor_speed_prev, MOTOR_TIME_CONSTANT_UP, MOTOR_TIME_CONSTANT_DOWN)
    motor_alpha = np.exp(-step_dt / np.maximum(motor_tau, 1.0e-9))
    return motor_alpha * motor_speed_prev + (1.0 - motor_alpha) * motor_speed_ref


def motor_speed_to_force_torque(motor_speed):
    rotor_forces = MOTOR_CONSTANT * motor_speed * motor_speed
    force_body = np.array([0.0, 0.0, -np.sum(rotor_forces)])
    torque_body = np.zeros(3)
    for motor_idx, (pos, force, yaw_sign) in enumerate(zip(ROTOR_POSITIONS, rotor_forces, ROTOR_YAW_SIGNS)):
        torque_body += np.cross(pos, np.array([0.0, 0.0, -force]))
        torque_body[2] += yaw_sign * MOMENT_CONSTANT * motor_speed[motor_idx] ** 2
    return rotor_forces, force_body, torque_body


def estimate_rate_int_bias_torque(rate_int, thrust_z_setpoint, step_dt):
    baseline_control = np.array([0.0, 0.0, 0.0, thrust_z_setpoint], dtype=float)
    trim_control = np.array([rate_int[0], rate_int[1], rate_int[2], thrust_z_setpoint], dtype=float)
    baseline_motor, _ = allocate_px4_quad_x(baseline_control)
    trim_motor, _ = allocate_px4_quad_x(trim_control)
    baseline_speed = motor_speed_from_setpoint(baseline_motor, step_dt)
    trim_speed = motor_speed_from_setpoint(trim_motor, step_dt)
    _, _, baseline_torque = motor_speed_to_force_torque(baseline_speed)
    _, _, trim_torque = motor_speed_to_force_torque(trim_speed)
    return trim_torque - baseline_torque


def should_apply_rate_int_bias_torque(using_actuator_motor_override):
    if not USE_RATE_INT_BIAS_TORQUE:
        return False
    return (not using_actuator_motor_override) or USE_RATE_INT_BIAS_TORQUE_WITH_ACTUATOR_LOG


def actuator_to_physical(actuator, q, omega_body, step_dt=DT, prev_motor_speed=None):
    actuator = np.clip(np.asarray(actuator, dtype=float)[:4], 0.0, 1.0)
    motor_speed = motor_speed_from_setpoint(actuator, step_dt, prev_motor_speed)
    rotor_forces, force_body, torque_body = motor_speed_to_force_torque(motor_speed)

    rot = quat_to_rotmat(q)
    acc_ned = rot @ force_body / DRONE_MASS + np.array([0.0, 0.0, A_OF_GRAVITY])
    inertia_omega = DRONE_INERTIA @ omega_body
    omega_dot = np.linalg.solve(DRONE_INERTIA, torque_body - np.cross(omega_body, inertia_omega))
    return {
        "motor_speed": motor_speed,
        "rotor_force": rotor_forces,
        "force_body": force_body,
        "torque_body": torque_body,
        "acc_ned": acc_ned,
        "omega_dot": omega_dot,
    }


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


def input_from_row(row, horizon_step=None, use_local_sp=False):
    def finite_row_value(name):
        if name in row.index and np.isfinite(row[name]):
            return row[name]
        return None

    if horizon_step is not None:
        x_ref = finite_row_value(f"u{horizon_step}_x")
        y_ref = finite_row_value(f"u{horizon_step}_y")
        z_ref = finite_row_value(f"u{horizon_step}_z")
        yaw_sp = finite_row_value(f"u{horizon_step}_yaw")
    else:
        x_ref = y_ref = z_ref = yaw_sp = None

    if x_ref is None and use_local_sp:
        x_ref = finite_row_value("local_sp_x")
    if x_ref is None:
        x_ref = finite_row_value("target_x_delayed")
    if x_ref is None:
        x_ref = finite_row_value("target_x")
    if x_ref is None:
        x_ref = row["pos_x"]

    if y_ref is None and use_local_sp:
        y_ref = finite_row_value("local_sp_y")
    if y_ref is None:
        y_ref = finite_row_value("target_y_delayed")
    if y_ref is None:
        y_ref = finite_row_value("target_y")
    if y_ref is None:
        y_ref = row["pos_y"]

    if z_ref is None and use_local_sp:
        z_ref = finite_row_value("local_sp_z")
    if z_ref is None:
        z_ref = finite_row_value("target_z_delayed")
    if z_ref is None:
        z_ref = finite_row_value("target_z")
    if z_ref is None:
        z_ref = row["pos_z"]

    if yaw_sp is None:
        yaw_sp = finite_row_value("target_yaw_delayed")
    if yaw_sp is None:
        yaw_sp = finite_row_value("target_yaw")
    if yaw_sp is None:
        yaw_sp = row["yaw"]

    pos_sp = np.array([
        x_ref,
        y_ref,
        z_ref,
    ], dtype=float)
    return pos_sp, yaw_sp


def apply_motor_command_delay(motor_setpoint, motor_command_delay_buffer, motor_command_delay_steps):
    motor_setpoint = np.asarray(motor_setpoint, dtype=float).copy()
    if motor_command_delay_buffer is None or motor_command_delay_steps <= 0:
        return motor_setpoint

    while len(motor_command_delay_buffer) < motor_command_delay_steps:
        motor_command_delay_buffer.append(motor_setpoint.copy())
    motor_command_delay_buffer.append(motor_setpoint.copy())
    return motor_command_delay_buffer.pop(0)


def actuator_motor_from_row(row):
    input_cols = [f"actuator_motor_input_{motor_idx}" for motor_idx in range(4)]
    actuator_cols = input_cols if all(col in row.index for col in input_cols) else [
        f"actuator_motor_{motor_idx}" for motor_idx in range(4)
    ]
    if not all(col in row.index for col in actuator_cols):
        return None
    actuator = row[actuator_cols].to_numpy(float)
    if not np.all(np.isfinite(actuator)):
        return None
    return np.clip(actuator, 0.0, 1.0)


def cu_mpc_simulator_step(
    s, pos_sp, yaw_sp, prev_vel, prev_acc, vel_int, step_dt,
    prev_omega=None, rate_int=None,
    z_vel_max_up=MPC_Z_VEL_MAX_UP, thrust_min=MPC_THR_MIN, no_thrust=False,
    hover_thrust=MPC_THR_HOVER, tilt_limit=MPC_TILT_MAX,
    rate_delay_buffer=None, rate_delay_steps=0,
    motor_command_delay_buffer=None,
    motor_command_delay_steps=MOTOR_COMMAND_DELAY_STEPS,
    landed_or_maybe_landed=False,
    saturation_positive=None, saturation_negative=None,
    yaw_torque_lpf_state=None,
    prev_motor_speed=None,
    prev_omega_dot=None,
    acceleration_bias=None,
    z_position_velocity_bias=0.0,
    actuator_motor_override=None,
):
    # s:機体状態vec 
    var = np.asarray(s, dtype=float).copy()
    q_prev = normalize(var[0:4])#現在姿勢
    if np.linalg.norm(q_prev) < 1.0e-8:
        q_prev = np.array([1.0, 0.0, 0.0, 0.0])
    var[0:4] = q_prev

    x_ref, y_ref, z_ref = np.asarray(pos_sp, dtype=float)
    yaw_ref = float(yaw_sp)
    if rate_int is None:
        rate_int = np.zeros(3)
    if acceleration_bias is None:
        acceleration_bias = np.zeros(3)
    else:
        acceleration_bias = np.asarray(acceleration_bias, dtype=float)

    # 現在速度
    vel_prev_state = var[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]].copy()
    # 現在速度の微分
    vel_dot = (vel_prev_state - prev_vel) / step_dt
    # LPF用の係数を計算 MPC_VELD_LP:速度微分LPFカットオフ周波数
    if MPC_VELD_LP > 1.0e-6:
        vel_dot_alpha = step_dt / (step_dt + 1.0 / (2.0 * np.pi * MPC_VELD_LP))
    else:
        vel_dot_alpha = 1.0
    # 速度微分にLPFを適用
    vel_dot_lpf = prev_acc + vel_dot_alpha * (vel_dot - prev_acc)
    # スラストを出さない状態・目標位置が無効な状態では、加速度目標値をPX4側でスラスト無効・未着陸に近い状態を表すダミー値にする
    if no_thrust or not np.all(np.isfinite([x_ref, y_ref, z_ref])):
        acc_setpoint = np.array([0.0, 0.0, 100.0])
        thrust_setpoint = np.zeros(3)
        torque_setpoint = np.zeros(3)
        motor_setpoint = np.zeros(4)
        if actuator_motor_override is not None and np.all(np.isfinite(actuator_motor_override)):
            motor_setpoint_used = np.clip(np.asarray(actuator_motor_override, dtype=float)[:4], 0.0, 1.0)
        else:
            motor_setpoint_used = apply_motor_command_delay(
                motor_setpoint,
                motor_command_delay_buffer,
                motor_command_delay_steps,
            )
        motor_speed = motor_speed_from_setpoint(motor_setpoint_used, step_dt, prev_motor_speed)
        rotor_forces = MOTOR_CONSTANT * motor_speed * motor_speed
        force_body = np.array([0.0, 0.0, -np.sum(rotor_forces)])
        torque_body = np.zeros(3)
        for motor_idx, (pos, force, yaw_sign) in enumerate(zip(ROTOR_POSITIONS, rotor_forces, ROTOR_YAW_SIGNS)):
            torque_body += np.cross(pos, np.array([0.0, 0.0, -force]))
            torque_body[2] += yaw_sign * MOMENT_CONSTANT * motor_speed[motor_idx] ** 2
        thrust_acc_ned = quat_to_rotmat(q_prev) @ force_body / DRONE_MASS + np.array([0.0, 0.0, A_OF_GRAVITY])
        translation_bias = acceleration_bias.copy()
        translation_bias[~TRANSLATION_BIAS_AXES] = 0.0
        acc_ned = thrust_acc_ned + translation_bias
        omega_dot_phys = np.linalg.solve(DRONE_INERTIA, torque_body)
        return var.copy(), vel_dot_lpf, np.zeros(3), {
            "vel_sp": np.full(3, np.nan),
            "acc_sp": acc_setpoint,
            "att_sp": np.array([1.0, 0.0, 0.0, 0.0]),
            "rate_sp": np.zeros(3),
            "rate_sp_used": np.zeros(3),
            "acc_sp_used": np.zeros(3),
            "z_velocity_bias": np.array([float(z_position_velocity_bias)]),
            "thrust_acc": thrust_acc_ned,
            "acc_used": acc_ned,
            "thr_sp": thrust_setpoint,
            "torque_sp": torque_setpoint,
            "torque_sp_unfiltered": torque_setpoint,
            "torque_p": np.zeros(3),
            "torque_i": np.zeros(3),
            "torque_d": np.zeros(3),
            "torque_ff": np.zeros(3),
            "motor_sp": motor_setpoint,
            "motor_from_px4_torque": motor_setpoint_used,
            "phys_acc": acc_ned,
            "phys_wdot": omega_dot_phys,
            "rate_derivative": np.zeros(3),
            "rate_derivative_next": np.zeros(3),
            "rate_int": np.zeros(3),
            "rate_int_next": np.zeros(3),
            "phys_force": force_body,
            "phys_torque": torque_body,
            "phys_torque_raw": torque_body,
            "bias_torque": np.zeros(3),
            "motor_speed": motor_speed,
            "allocator_unallocated_torque": np.zeros(3),
            "allocator_saturation_positive": np.zeros(3, dtype=bool),
            "allocator_saturation_negative": np.zeros(3, dtype=bool),
        }
    # 位置制御のP制御で速度目標値を計算
    vel_setpoint = np.array([
        MPC_XY_P * (x_ref - var[STATE_INDEX["x"]]),
        MPC_XY_P * (y_ref - var[STATE_INDEX["y"]]),
        MPC_Z_P * (z_ref - var[STATE_INDEX["z"]]),
    ])
    # 水平速度の制限
    vel_xy_norm = np.sqrt(vel_setpoint[0] ** 2 + vel_setpoint[1] ** 2)
    if vel_xy_norm > MPC_XY_VEL_MAX and vel_xy_norm > 1.0e-8:
        vel_setpoint[:2] = vel_setpoint[:2] / vel_xy_norm * MPC_XY_VEL_MAX
    z_vel_min = -z_vel_max_up
    if vel_setpoint[2] < z_vel_min:
        vel_setpoint[2] = z_vel_min
    elif vel_setpoint[2] > MPC_Z_VEL_MAX_DOWN:
        vel_setpoint[2] = MPC_Z_VEL_MAX_DOWN
    # 速度誤差を計算
    vel_error = vel_setpoint - vel_prev_state
    # 加速度目標値を計算
    acc_setpoint = np.array([
        MPC_XY_VEL_P_ACC * vel_error[0] + vel_int[0] - MPC_XY_VEL_D_ACC * vel_dot_lpf[0],
        MPC_XY_VEL_P_ACC * vel_error[1] + vel_int[1] - MPC_XY_VEL_D_ACC * vel_dot_lpf[1],
        MPC_Z_VEL_P_ACC * vel_error[2] + vel_int[2] - MPC_Z_VEL_D_ACC * vel_dot_lpf[2],
    ])
    # 加速度目標値から目標機体姿勢のz軸を計算
    z_specific_force = A_OF_GRAVITY if MPC_ACC_DECOUPLE else A_OF_GRAVITY - acc_setpoint[2]
    body_z = np.array([-acc_setpoint[0], -acc_setpoint[1], z_specific_force])
    body_z = normalize(body_z)
    if np.linalg.norm(body_z) < 1.0e-8:
        body_z = np.array([0.0, 0.0, 1.0])
    # 目標姿勢body_zが鉛直方向からどれだけ傾いているか
    tilt_angle = np.arccos(np.clip(body_z[2], -1.0, 1.0))
    # 傾き方向を維持した状態で角度はtilt_limitに制限
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
    # 目標z軸の鉛直軸に対するcos((0, 0, 1)との内積)
    cos_ned_body = body_z[2] if abs(body_z[2]) >= 1.0e-6 else 1.0e-6
    # スラストを目標z軸方向に変換 thrust_ned_z = collective_thrust*cos_ned_body
    collective_thrust = min(thrust_ned_z / cos_ned_body, -thrust_min)
    # z軸方向にcollective_thrustのスラスト目標値を生成
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

    # 制限後のxyスラストから生成される加速度目標値を計算
    acc_sp_xy_produced = thrust_setpoint[:2] * (A_OF_GRAVITY / hover_thrust)
    # xy方向アンチワインドアップ補正
    if np.dot(acc_setpoint[:2], acc_setpoint[:2]) > np.dot(acc_sp_xy_produced, acc_sp_xy_produced):
        arw_gain = 2.0 / MPC_XY_VEL_P_ACC
        vel_error_for_int[:2] -= arw_gain * (acc_setpoint[:2] - acc_sp_xy_produced)
    vel_error_for_int[~np.isfinite(vel_error_for_int)] = 0.0
    thrust_body_setpoint = np.array([0.0, 0.0, -np.linalg.norm(thrust_setpoint)], dtype=float)

    att_body_z = normalize(-thrust_setpoint)
    if np.linalg.norm(att_body_z) < 1.0e-8:
        att_body_z = np.array([0.0, 0.0, 1.0])
    sy, cy = sin_cos_simulator(yaw_ref)
    # yaw目標値のsin,cosから水平面内の基準y軸を作成
    y_c = np.array([-sy, cy, 0.0])
    # y_cと目標z軸の外積から目標x軸を計算
    body_x = np.cross(y_c, att_body_z)
    if att_body_z[2] < 0.0:
        body_x = -body_x
    if abs(att_body_z[2]) < 1.0e-6:
        body_x = np.array([0.0, 0.0, 1.0])
    body_x = normalize(body_x)
    body_y = np.cross(att_body_z, body_x)
    att_setpoint = rotmat_to_quat(body_x, body_y, att_body_z)

    qd = att_setpoint.copy()
    e_z = quat_to_rotmat(q_prev)[:, 2]
    e_z_d = quat_to_rotmat(qd)[:, 2]
    tilt_axis = np.cross(e_z, e_z_d)
    tilt_axis_norm = np.linalg.norm(tilt_axis)
    tilt_dot = np.clip(np.dot(e_z, e_z_d), -1.0, 1.0)
    if tilt_axis_norm < 1.0e-8:
        qd_red = np.array([1.0, 0.0, 0.0, 0.0]) if tilt_dot > 0.0 else qd.copy()
    else:
        tilt_axis /= tilt_axis_norm
        tilt_angle = np.arctan2(tilt_axis_norm, tilt_dot)
        qd_red = normalize(np.array([
            np.cos(0.5 * tilt_angle),
            tilt_axis[0] * np.sin(0.5 * tilt_angle),
            tilt_axis[1] * np.sin(0.5 * tilt_angle),
            tilt_axis[2] * np.sin(0.5 * tilt_angle),
        ]))
        if abs(qd_red[1]) > 1.0 - 1.0e-5 or abs(qd_red[2]) > 1.0 - 1.0e-5:
            qd_red = qd.copy()
        else:
            a0, a1, a2, a3 = qd_red
            b0, b1, b2, b3 = q_prev
            qd_red = normalize(np.array([
                a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
            ]))

    a0, a1, a2, a3 = np.array([qd_red[0], -qd_red[1], -qd_red[2], -qd_red[3]])
    b0, b1, b2, b3 = qd
    qd_dyaw = normalize(np.array([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ]))
    if qd_dyaw[0] < 0.0:
        qd_dyaw = -qd_dyaw
    qd_dyaw[0] = np.clip(qd_dyaw[0], -1.0, 1.0)
    qd_dyaw[3] = np.clip(qd_dyaw[3], -1.0, 1.0)
    q_yaw_weighted = np.array([
        np.cos(MC_YAW_WEIGHT * np.arccos(qd_dyaw[0])),
        0.0,
        0.0,
        np.sin(MC_YAW_WEIGHT * np.arcsin(qd_dyaw[3])),
    ])
    a0, a1, a2, a3 = qd_red
    b0, b1, b2, b3 = q_yaw_weighted
    qd_weighted = normalize(np.array([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ]))
    a0, a1, a2, a3 = np.array([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]])
    b0, b1, b2, b3 = qd_weighted
    qe = normalize(np.array([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ]))
    if qe[0] < 0.0:
        qe = -qe
    attitude_gain = np.array([
        MC_ROLL_P,
        MC_PITCH_P,
        MC_YAW_P / MC_YAW_WEIGHT if MC_YAW_WEIGHT > 1.0e-4 else MC_YAW_P,
    ])
    omega_setpoint = 2.0 * qe[1:4] * attitude_gain
    omega_setpoint = np.clip(
        omega_setpoint,
        -np.array([MC_ROLLRATE_MAX, MC_PITCHRATE_MAX, MC_YAWRATE_MAX]),
        np.array([MC_ROLLRATE_MAX, MC_PITCHRATE_MAX, MC_YAWRATE_MAX]),
    )

    v_prev = vel_prev_state.copy()
    w_prev = var[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]].copy()
    if prev_omega is None:
        prev_omega = w_prev.copy()
    if prev_omega_dot is None:
        prev_omega_dot = np.zeros(3)
    raw_omega_dot = (w_prev - prev_omega) / step_dt
    if ANGULAR_ACCEL_LP > 1.0e-6:
        omega_dot_alpha = step_dt / (step_dt + 1.0 / (2.0 * np.pi * ANGULAR_ACCEL_LP))
        omega_dot = prev_omega_dot + omega_dot_alpha * (raw_omega_dot - prev_omega_dot)
    else:
        omega_dot = raw_omega_dot
    rate_error = omega_setpoint - w_prev
    rate_p = np.array([
        MC_ROLLRATE_K * MC_ROLLRATE_P,
        MC_PITCHRATE_K * MC_PITCHRATE_P,
        MC_YAWRATE_K * MC_YAWRATE_P,
    ])
    rate_i = np.array([
        MC_ROLLRATE_K * MC_ROLLRATE_I,
        MC_PITCHRATE_K * MC_PITCHRATE_I,
        MC_YAWRATE_K * MC_YAWRATE_I,
    ])
    rate_d = np.array([
        MC_ROLLRATE_K * MC_ROLLRATE_D,
        MC_PITCHRATE_K * MC_PITCHRATE_D,
        MC_YAWRATE_K * MC_YAWRATE_D,
    ])
    rate_ff = np.array([
        MC_ROLLRATE_FF,
        MC_PITCHRATE_FF,
        MC_YAWRATE_FF,
    ])
    rate_int_lim = np.array([MC_RR_INT_LIM, MC_PR_INT_LIM, MC_YR_INT_LIM])
    saturation_positive = (
        np.zeros(3, dtype=bool)
        if saturation_positive is None
        else np.asarray(saturation_positive, dtype=bool)
    )
    saturation_negative = (
        np.zeros(3, dtype=bool)
        if saturation_negative is None
        else np.asarray(saturation_negative, dtype=bool)
    )

    torque_p = rate_p * rate_error
    torque_i = rate_int.copy()
    torque_d = -rate_d * omega_dot
    torque_ff = rate_ff * omega_setpoint
    torque_setpoint_unfiltered = torque_p + torque_i + torque_d + torque_ff
    torque_setpoint = torque_setpoint_unfiltered.copy()
    if MC_YAW_TQ_CUTOFF > 1.0e-6:
        yaw_alpha = step_dt / (step_dt + 1.0 / (2.0 * np.pi * MC_YAW_TQ_CUTOFF))
        yaw_torque_prev = torque_setpoint[2] if yaw_torque_lpf_state is None else float(yaw_torque_lpf_state)
        torque_setpoint[2] = yaw_torque_prev + yaw_alpha * (torque_setpoint[2] - yaw_torque_prev)
        yaw_torque_lpf_next = torque_setpoint[2]
    else:
        yaw_torque_lpf_next = torque_setpoint[2]

    rate_int_next = rate_int.copy()
    if not landed_or_maybe_landed:
        for axis in range(3):
            rate_error_for_int = rate_error[axis]
            if saturation_positive[axis]:
                rate_error_for_int = min(rate_error_for_int, 0.0)
            if saturation_negative[axis]:
                rate_error_for_int = max(rate_error_for_int, 0.0)

            i_factor = rate_error_for_int / np.deg2rad(400.0)
            i_factor = max(0.0, 1.0 - i_factor * i_factor)
            rate_int_next[axis] += i_factor * rate_i[axis] * rate_error_for_int * step_dt
            if np.isfinite(rate_int_next[axis]):
                rate_int_next[axis] = np.clip(rate_int_next[axis], -rate_int_lim[axis], rate_int_lim[axis])
            else:
                rate_int_next[axis] = rate_int[axis]

    control_sp = np.array([
        torque_setpoint[0],
        torque_setpoint[1],
        torque_setpoint[2],
        thrust_body_setpoint[2],
    ], dtype=float)
    motor_setpoint, unallocated_control = allocate_px4_quad_x(control_sp)
    allocator_saturation_positive_next = unallocated_control[:3] > np.finfo(float).eps
    allocator_saturation_negative_next = unallocated_control[:3] < -np.finfo(float).eps

    using_actuator_motor_override = (
        actuator_motor_override is not None
        and np.all(np.isfinite(actuator_motor_override))
    )
    if using_actuator_motor_override:
        motor_setpoint_used = np.clip(np.asarray(actuator_motor_override, dtype=float)[:4], 0.0, 1.0)
    else:
        motor_setpoint_used = apply_motor_command_delay(
            motor_setpoint,
            motor_command_delay_buffer,
            motor_command_delay_steps,
        )
    motor_speed = motor_speed_from_setpoint(motor_setpoint_used, step_dt, prev_motor_speed)
    rotor_forces, force_body, torque_body_raw = motor_speed_to_force_torque(motor_speed)
    if should_apply_rate_int_bias_torque(using_actuator_motor_override):
        bias_torque = estimate_rate_int_bias_torque(rate_int, thrust_body_setpoint[2], step_dt)
        torque_body = torque_body_raw - bias_torque
    else:
        bias_torque = np.zeros(3)
        torque_body = torque_body_raw
    thrust_acc_ned = quat_to_rotmat(q_prev) @ force_body / DRONE_MASS + np.array([0.0, 0.0, A_OF_GRAVITY])
    translation_bias = acceleration_bias.copy()
    translation_bias[~TRANSLATION_BIAS_AXES] = 0.0
    acc_ned = thrust_acc_ned + translation_bias
    inertia_omega = DRONE_INERTIA @ w_prev
    omega_dot_phys = np.linalg.solve(DRONE_INERTIA, torque_body - np.cross(w_prev, inertia_omega))
    acc_for_dynamics = acc_ned.copy()

    omega_dot_for_dynamics = omega_dot_phys.copy()
    if rate_delay_buffer is not None and rate_delay_steps > 0:
        while len(rate_delay_buffer) < rate_delay_steps:
            rate_delay_buffer.append(np.zeros(3))
        rate_delay_buffer.append(omega_dot_for_dynamics.copy())
        omega_dot_for_dynamics = rate_delay_buffer.pop(0)

    w_next = w_prev + omega_dot_for_dynamics * step_dt
    w_mid = 0.5 * (w_prev + w_next)

    next_var = var.copy()
    next_var[STATE_INDEX["x"]] += v_prev[0] * step_dt
    next_var[STATE_INDEX["y"]] += v_prev[1] * step_dt
    z_velocity_for_position = v_prev[2] - float(z_position_velocity_bias)
    next_var[STATE_INDEX["z"]] += z_velocity_for_position * step_dt
    next_var[STATE_INDEX["vx"]] = v_prev[0] + acc_for_dynamics[0] * step_dt
    next_var[STATE_INDEX["vy"]] = v_prev[1] + acc_for_dynamics[1] * step_dt
    next_var[STATE_INDEX["vz"]] = v_prev[2] + acc_for_dynamics[2] * step_dt

    q_dot = np.array([
        -0.5 * (q_prev[1] * w_mid[0] + q_prev[2] * w_mid[1] + q_prev[3] * w_mid[2]),
        0.5 * (q_prev[0] * w_mid[0] + q_prev[2] * w_mid[2] - q_prev[3] * w_mid[1]),
        0.5 * (q_prev[0] * w_mid[1] - q_prev[1] * w_mid[2] + q_prev[3] * w_mid[0]),
        0.5 * (q_prev[0] * w_mid[2] + q_prev[1] * w_mid[1] - q_prev[2] * w_mid[0]),
    ])
    next_var[0:4] = normalize(q_prev + q_dot * step_dt)
    next_var[STATE_INDEX["wx"]] = w_next[0]
    next_var[STATE_INDEX["wy"]] = w_next[1]
    next_var[STATE_INDEX["wz"]] = w_next[2]

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
        "rate_sp_used": w_next,
        "acc_sp_used": acc_for_dynamics,
        "thrust_acc": thrust_acc_ned,
        "acc_used": acc_for_dynamics,
        "z_velocity_bias": np.array([float(z_position_velocity_bias)]),
        "thr_sp": thrust_body_setpoint,
        "torque_sp": torque_setpoint,
        "torque_sp_unfiltered": torque_setpoint_unfiltered,
        "torque_p": torque_p,
        "torque_i": torque_i,
        "torque_d": torque_d,
        "torque_ff": torque_ff,
        "motor_sp": motor_setpoint,
        "motor_from_px4_torque": motor_setpoint_used,
        "phys_acc": acc_ned,
        "phys_wdot": omega_dot_phys,
        "rate_derivative": omega_dot,
        "rate_derivative_next": omega_dot,
        "rate_int": rate_int,
        "phys_force": force_body,
        "phys_torque": torque_body,
        "phys_torque_raw": torque_body_raw,
        "bias_torque": bias_torque,
        "motor_speed": motor_speed,
        "rate_int_next": rate_int_next,
        "yaw_torque_lpf_next": yaw_torque_lpf_next,
        "allocator_unallocated_torque": unallocated_control[:3],
        "allocator_saturation_positive": allocator_saturation_positive_next,
        "allocator_saturation_negative": allocator_saturation_negative_next,
    }


RATE_INT_COLUMNS = ["rollspeed_integ", "pitchspeed_integ", "yawspeed_integ"]


def logged_rate_int_from_row(row):
    if all(col in row.index for col in RATE_INT_COLUMNS):
        logged_rate_int = row[RATE_INT_COLUMNS].to_numpy(float)
        if np.all(np.isfinite(logged_rate_int)):
            return logged_rate_int
    return None


def estimate_vel_int_from_logged_setpoint(row, state, prev_acc):
    required = ["local_sp_vx", "local_sp_vy", "local_sp_vz", "local_sp_ax", "local_sp_ay", "local_sp_az"]
    if not all(col in row.index for col in required):
        return None

    local_sp_vel = row[["local_sp_vx", "local_sp_vy", "local_sp_vz"]].to_numpy(float)
    local_sp_acc = row[["local_sp_ax", "local_sp_ay", "local_sp_az"]].to_numpy(float)
    if not np.all(np.isfinite(local_sp_vel)) or not np.all(np.isfinite(local_sp_acc)):
        return None
    if abs(local_sp_acc[2]) > 50.0:
        return None

    current_vel = state[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]]
    vel_error = local_sp_vel - current_vel
    vel_int = np.array([
        local_sp_acc[0] - MPC_XY_VEL_P_ACC * vel_error[0] + MPC_XY_VEL_D_ACC * prev_acc[0],
        local_sp_acc[1] - MPC_XY_VEL_P_ACC * vel_error[1] + MPC_XY_VEL_D_ACC * prev_acc[1],
        local_sp_acc[2] - MPC_Z_VEL_P_ACC * vel_error[2] + MPC_Z_VEL_D_ACC * prev_acc[2],
    ])
    vel_int[2] = np.clip(vel_int[2], -A_OF_GRAVITY, A_OF_GRAVITY)
    return vel_int


def z_position_velocity_bias_observation(row):
    z_deriv_col = None
    for candidate in ("local_pos_z_deriv", "z_deriv"):
        if candidate in row.index:
            z_deriv_col = candidate
            break
    if z_deriv_col is None or "vel_z" not in row.index:
        return None
    vel_z = float(row["vel_z"])
    z_deriv = float(row[z_deriv_col])
    if not np.isfinite(vel_z) or not np.isfinite(z_deriv):
        return None
    return vel_z - z_deriv


def build_history(
    df,
    step_dt,
    rate_delay_steps=RATE_DELAY_STEPS,
    motor_command_delay_steps=MOTOR_COMMAND_DELAY_STEPS,
    rate_int_sync_period=RATE_INT_SYNC_PERIOD,
    return_context=False,
    use_local_sp_position=False,
    motor_input_source="model",
):
    n = len(df)
    progress(f"build_history start: {n} samples")
    history_progress_stride = max(1, min(50, n // 10 if n >= 10 else 1))
    pred_next = np.full((n, len(STATE_NAMES)), np.nan)
    actual = np.full((n, len(STATE_NAMES)), np.nan)
    actual_next = np.full((n, len(STATE_NAMES)), np.nan)
    calc = {
        "vel_sp": np.full((n, 3), np.nan),
        "acc_sp": np.full((n, 3), np.nan),
        "att_sp": np.full((n, 4), np.nan),
        "rate_sp": np.full((n, 3), np.nan),
        "thr_sp": np.full((n, 3), np.nan),
        "torque_sp": np.full((n, 3), np.nan),
        "torque_sp_unfiltered": np.full((n, 3), np.nan),
        "torque_p": np.full((n, 3), np.nan),
        "torque_i": np.full((n, 3), np.nan),
        "torque_d": np.full((n, 3), np.nan),
        "torque_ff": np.full((n, 3), np.nan),
        "motor_sp": np.full((n, 4), np.nan),
        "motor_from_px4_torque": np.full((n, 4), np.nan),
        "phys_acc": np.full((n, 3), np.nan),
        "thrust_acc": np.full((n, 3), np.nan),
        "acc_used": np.full((n, 3), np.nan),
        "actuator_acc": np.full((n, 3), np.nan),
        "acc_residual": np.full((n, 3), np.nan),
        "phys_wdot": np.full((n, 3), np.nan),
        "rate_derivative": np.full((n, 3), np.nan),
        "rate_int": np.full((n, 3), np.nan),
        "allocator_unallocated_torque": np.full((n, 3), np.nan),
        "phys_force": np.full((n, 3), np.nan),
        "phys_torque": np.full((n, 3), np.nan),
        "phys_torque_raw": np.full((n, 3), np.nan),
        "bias_torque": np.full((n, 3), np.nan),
        "motor_speed": np.full((n, 4), np.nan),
        "acceleration_bias": np.full((n, 3), np.nan),
        "z_velocity_bias": np.full((n, 1), np.nan),
        "imu_acc": np.full((n, 3), np.nan),
    }

    prev_vel = df.loc[0, ["vel_x", "vel_y", "vel_z"]].to_numpy(float)
    prev_omega = df.loc[0, ["angular_vel_x", "angular_vel_y", "angular_vel_z"]].to_numpy(float)
    prev_omega_dot = np.zeros(3)
    prev_acc = np.zeros(3)
    vel_int = np.zeros(3)
    rate_int = np.zeros(3)
    logged_rate_int = logged_rate_int_from_row(df.loc[0])
    if logged_rate_int is not None:
        rate_int = logged_rate_int.copy()
    next_rate_int_sync_time = (
        float(df.loc[0, "control_time_s"]) + rate_int_sync_period
        if rate_int_sync_period and rate_int_sync_period > 0.0
        else np.inf
    )
    motor_speed = np.zeros(4)
    hover_thrust = MPC_THR_HOVER
    last_hover_thrust_log = np.nan
    prev_model_acc = np.full(3, np.nan)
    rate_delay_buffer = []
    motor_command_delay_buffer = []
    yaw_torque_lpf_state = None
    allocator_saturation_positive = np.zeros(3, dtype=bool)
    allocator_saturation_negative = np.zeros(3, dtype=bool)
    acceleration_bias = np.zeros(3)
    z_position_velocity_bias = 0.0
    takeoff_state = TAKEOFF_STATE_FLIGHT
    context_hist = None
    if return_context:
        context_hist = {
            "prev_vel": np.full((n, 3), np.nan),
            "prev_acc": np.full((n, 3), np.nan),
            "vel_int": np.full((n, 3), np.nan),
            "prev_omega": np.full((n, 3), np.nan),
            "prev_omega_dot": np.full((n, 3), np.nan),
            "rate_int": np.full((n, 3), np.nan),
            "motor_speed": np.full((n, 4), np.nan),
            "yaw_torque_lpf_state": np.full(n, np.nan),
            "allocator_saturation_positive": np.zeros((n, 3), dtype=bool),
            "allocator_saturation_negative": np.zeros((n, 3), dtype=bool),
            "rate_delay_buffer": [],
            "motor_command_delay_buffer": [],
            "hover_thrust": np.full(n, np.nan),
            "z_vel_max_up": np.full(n, np.nan),
            "thrust_min": np.full(n, np.nan),
            "no_thrust": np.zeros(n, dtype=bool),
            "tilt_limit": np.full(n, np.nan),
            "landed_or_maybe_landed": np.zeros(n, dtype=bool),
            "acceleration_bias": np.full((n, 3), np.nan),
            "z_velocity_bias": np.full(n, np.nan),
            "acceleration_bias_observed": np.full((n, 3), np.nan),
        }

    for i in range(n):
        if i == 0 or (i + 1) % history_progress_stride == 0 or i + 1 == n:
            row_time = float(df.iloc[i]["control_time_s"])
            progress(
                f"build_history {i + 1}/{n} "
                f"({100.0 * (i + 1) / max(n, 1):.0f}%, t={row_time:.3f}s)"
            )
        row = df.iloc[i]
        if i > 0 and "_segment_id" in df.columns and row["_segment_id"] != df.iloc[i - 1]["_segment_id"]:
            vel_int = np.zeros(3)
            logged_rate_int = logged_rate_int_from_row(row)
            if logged_rate_int is not None:
                rate_int = logged_rate_int.copy()
                next_rate_int_sync_time = (
                    float(row["control_time_s"]) + rate_int_sync_period
                    if rate_int_sync_period and rate_int_sync_period > 0.0
                    else np.inf
                )
            z_position_velocity_bias = 0.0
            motor_command_delay_buffer = []
        s = state_from_row(row)
        actual[i] = s
        if i + 1 < n:
            actual_next[i] = state_from_row(df.iloc[i + 1])
        imu_acc = (
            row[["livox_imu_acc_x", "livox_imu_acc_y", "livox_imu_acc_z"]].to_numpy(float)
            if all(col in row.index for col in ["livox_imu_acc_x", "livox_imu_acc_y", "livox_imu_acc_z"])
            else np.full(3, np.nan)
        )
        if np.all(np.isfinite(imu_acc)):
            calc["imu_acc"][i] = imu_acc

        if rate_int_sync_period and rate_int_sync_period > 0.0:
            row_time = float(row["control_time_s"])
            if row_time + 0.5 * step_dt >= next_rate_int_sync_time:
                logged_rate_int = logged_rate_int_from_row(row)
                if logged_rate_int is not None:
                    rate_int = logged_rate_int.copy()
                sync_deadline = row_time + 0.5 * step_dt
                if np.isfinite(next_rate_int_sync_time):
                    missed_syncs = np.floor(
                        (sync_deadline - next_rate_int_sync_time) / rate_int_sync_period
                    ) + 1.0
                    next_rate_int_sync_time += max(1.0, missed_syncs) * rate_int_sync_period
                else:
                    next_rate_int_sync_time = sync_deadline + rate_int_sync_period

        pos_sp, yaw_sp = input_from_row(row, use_local_sp=use_local_sp_position)
        if "hover_thrust" in row.index and "hover_thrust_valid" in row.index:
            hover_thrust_new = float(row["hover_thrust"])
            hover_thrust_valid = bool(np.isfinite(row["hover_thrust_valid"]) and row["hover_thrust_valid"] != 0.0)
            hover_thrust_updated = (
                not np.isfinite(last_hover_thrust_log)
                or abs(hover_thrust_new - last_hover_thrust_log) > 1.0e-7
            )
            if hover_thrust_valid and hover_thrust_updated and np.isfinite(hover_thrust_new) and hover_thrust_new > 1.0e-6:
                hover_thrust = hover_thrust_new
                last_hover_thrust_log = hover_thrust_new

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

        not_taken_off = takeoff_state < TAKEOFF_STATE_RAMPUP
        flying = takeoff_state >= TAKEOFF_STATE_FLIGHT
        flying_but_ground_contact = flying and (ground_contact or maybe_landed)
        z_vel_max_up = MPC_Z_VEL_MAX_UP
        thrust_min = MPC_THR_MIN if flying else 0.0

        logged_vel_int = estimate_vel_int_from_logged_setpoint(row, s, prev_acc)
        if logged_vel_int is not None:
            vel_int = logged_vel_int

        if (
            USE_ACCELERATION_BIAS_OBSERVER
            and USE_VELOCITY_DELTA_BIAS_OBSERVER
            and i > 0
            and np.any(VELOCITY_DELTA_BIAS_AXES)
        ):
            logged_acc = (
                s[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]]
                - prev_vel
            ) / step_dt
            if np.all(np.isfinite(logged_acc)) and np.all(np.isfinite(prev_model_acc)):
                residual = logged_acc - prev_model_acc
                residual[np.abs(residual) < ACCELERATION_BIAS_DEADBAND] = 0.0
                axes = VELOCITY_DELTA_BIAS_AXES
                acceleration_bias[axes] = (
                    acceleration_bias[axes]
                    + VELOCITY_DELTA_BIAS_ALPHA * residual[axes]
                )
                acceleration_bias = np.clip(
                    acceleration_bias,
                    -ACCELERATION_BIAS_LIMIT,
                    ACCELERATION_BIAS_LIMIT,
                )

        if return_context:
            context_hist["prev_vel"][i] = prev_vel.copy()
            context_hist["prev_acc"][i] = prev_acc.copy()
            context_hist["vel_int"][i] = vel_int.copy()
            context_hist["prev_omega"][i] = prev_omega.copy()
            context_hist["prev_omega_dot"][i] = prev_omega_dot.copy()
            context_hist["rate_int"][i] = rate_int.copy()
            context_hist["motor_speed"][i] = motor_speed.copy()
            if yaw_torque_lpf_state is not None and np.isfinite(yaw_torque_lpf_state):
                context_hist["yaw_torque_lpf_state"][i] = float(yaw_torque_lpf_state)
            context_hist["allocator_saturation_positive"][i] = allocator_saturation_positive.copy()
            context_hist["allocator_saturation_negative"][i] = allocator_saturation_negative.copy()
            context_hist["rate_delay_buffer"].append([entry.copy() for entry in rate_delay_buffer])
            context_hist["motor_command_delay_buffer"].append([entry.copy() for entry in motor_command_delay_buffer])
            context_hist["hover_thrust"][i] = hover_thrust
            context_hist["z_vel_max_up"][i] = z_vel_max_up
            context_hist["thrust_min"][i] = thrust_min
            context_hist["no_thrust"][i] = bool(not_taken_off or flying_but_ground_contact)
            context_hist["tilt_limit"][i] = tilt_limit
            context_hist["landed_or_maybe_landed"][i] = bool(landed or maybe_landed)
            context_hist["acceleration_bias"][i] = acceleration_bias.copy()
            context_hist["z_velocity_bias"][i] = z_position_velocity_bias

        s_next, prev_acc, vel_int, debug = cu_mpc_simulator_step(
            s, pos_sp, yaw_sp, prev_vel, prev_acc, vel_int, step_dt,
            prev_omega=prev_omega, rate_int=rate_int,
            z_vel_max_up=z_vel_max_up, thrust_min=thrust_min,
            no_thrust=(not_taken_off or flying_but_ground_contact),
            hover_thrust=hover_thrust, tilt_limit=tilt_limit,
            rate_delay_buffer=rate_delay_buffer,
            rate_delay_steps=rate_delay_steps,
            motor_command_delay_buffer=motor_command_delay_buffer,
            motor_command_delay_steps=motor_command_delay_steps,
            landed_or_maybe_landed=(landed or maybe_landed),
            saturation_positive=allocator_saturation_positive,
            saturation_negative=allocator_saturation_negative,
            yaw_torque_lpf_state=yaw_torque_lpf_state,
            prev_motor_speed=motor_speed,
            prev_omega_dot=prev_omega_dot,
            acceleration_bias=acceleration_bias,
            z_position_velocity_bias=(
                z_position_velocity_bias
                if USE_Z_POSITION_VELOCITY_BIAS_CORRECTION
                else 0.0
            ),
            actuator_motor_override=(
                actuator_motor_from_row(row)
                if motor_input_source == "actuator_log"
                else None
            ),
        )
        rate_int = debug.get("rate_int_next", rate_int)
        prev_omega_dot = debug.get("rate_derivative_next", prev_omega_dot)
        motor_speed = debug.get("motor_speed", motor_speed)
        yaw_torque_lpf_state = debug.get("yaw_torque_lpf_next", yaw_torque_lpf_state)
        allocator_saturation_positive = debug.get("allocator_saturation_positive", allocator_saturation_positive)
        allocator_saturation_negative = debug.get("allocator_saturation_negative", allocator_saturation_negative)
        pred_next[i] = s_next
        for key in calc:
            if key in debug:
                calc[key][i] = debug[key]
        calc["acceleration_bias"][i] = acceleration_bias.copy()
        calc["z_velocity_bias"][i, 0] = (
            z_position_velocity_bias
            if USE_Z_POSITION_VELOCITY_BIAS_CORRECTION
            else 0.0
        )
        actuator = actuator_motor_from_row(row)
        if actuator is not None:
            actuator_physical = actuator_to_physical(
                actuator,
                s[0:4],
                s[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]],
                step_dt=step_dt,
            )
            calc["actuator_acc"][i] = actuator_physical["acc_ned"]
        observed_acceleration_bias = acceleration_bias.copy()
        modeled_acc = debug.get("acc_sp_used", debug.get("phys_acc", np.full(3, np.nan)))
        if np.all(np.isfinite(imu_acc)) and np.all(np.isfinite(modeled_acc)):
            calc["acc_residual"][i] = imu_acc - modeled_acc
        if np.all(np.isfinite(imu_acc)) and np.all(np.isfinite(modeled_acc)):
            observed_acceleration_bias = imu_acc - (modeled_acc - acceleration_bias)
            observed_acceleration_bias = np.clip(
                observed_acceleration_bias,
                -ACCELERATION_BIAS_LIMIT,
                ACCELERATION_BIAS_LIMIT,
            )
            if USE_VELOCITY_DELTA_BIAS_OBSERVER and np.any(VELOCITY_DELTA_BIAS_AXES):
                observed_acceleration_bias[VELOCITY_DELTA_BIAS_AXES] = acceleration_bias[
                    VELOCITY_DELTA_BIAS_AXES
                ]
        if return_context:
            context_hist["acceleration_bias_observed"][i] = observed_acceleration_bias.copy()
        if USE_ACCELERATION_BIAS_OBSERVER and i + 1 < n and np.all(np.isfinite(actual_next[i])):
            if USE_LIVOX_IMU_ACCELERATION and np.all(np.isfinite(imu_acc)):
                actual_acc = imu_acc
            else:
                actual_acc = (
                    actual_next[i][[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]]
                    - s[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]]
                ) / step_dt
            if np.all(np.isfinite(actual_acc)) and np.all(np.isfinite(modeled_acc)):
                residual = actual_acc - modeled_acc
                if USE_VELOCITY_DELTA_BIAS_OBSERVER and np.any(VELOCITY_DELTA_BIAS_AXES):
                    residual[VELOCITY_DELTA_BIAS_AXES] = 0.0
                residual[np.abs(residual) < ACCELERATION_BIAS_DEADBAND] = 0.0
                acceleration_bias = acceleration_bias + ACCELERATION_BIAS_ALPHA * residual
                acceleration_bias = np.clip(
                    acceleration_bias,
                    -ACCELERATION_BIAS_LIMIT,
                    ACCELERATION_BIAS_LIMIT,
                )
        observed_z_velocity_bias = z_position_velocity_bias_observation(row)
        if USE_Z_POSITION_VELOCITY_BIAS_CORRECTION and observed_z_velocity_bias is not None:
            z_position_velocity_bias += (
                Z_POSITION_VELOCITY_BIAS_ALPHA
                * (observed_z_velocity_bias - z_position_velocity_bias)
            )
            z_position_velocity_bias = float(np.clip(
                z_position_velocity_bias,
                -Z_POSITION_VELOCITY_BIAS_LIMIT,
                Z_POSITION_VELOCITY_BIAS_LIMIT,
            ))
        prev_vel = s[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]].copy()
        prev_model_acc = debug.get("acc_used", prev_model_acc).copy()
        prev_omega = s[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]].copy()

    if return_context:
        progress("build_history done")
        return actual, actual_next, pred_next, calc, context_hist

    progress("build_history done")
    return actual, actual_next, pred_next, calc


def finite_vector(value, fallback):
    value = np.asarray(value, dtype=float)
    if value.shape == np.asarray(fallback).shape and np.all(np.isfinite(value)):
        return value.copy()
    return np.asarray(fallback, dtype=float).copy()


def finite_scalar(value, fallback):
    value = float(value)
    if np.isfinite(value):
        return value
    return fallback


def selected_prediction_start_indices(plot_t, active_mask, prediction_interval):
    active_indices = np.flatnonzero(active_mask)
    if len(active_indices) == 0:
        return active_indices
    if prediction_interval <= 0.0:
        return active_indices

    starts = [int(active_indices[0])]
    next_start_time = float(plot_t[active_indices[0]]) + prediction_interval
    for idx in active_indices[1:]:
        if float(plot_t[idx]) + 1.0e-9 >= next_start_time:
            starts.append(int(idx))
            next_start_time = float(plot_t[idx]) + prediction_interval
    return np.asarray(starts, dtype=int)


def build_multistep_predictions(
    df,
    actual,
    context_hist,
    start_indices,
    step_dt,
    horizon,
    rate_delay_steps=RATE_DELAY_STEPS,
    motor_command_delay_steps=MOTOR_COMMAND_DELAY_STEPS,
    use_local_sp_position=False,
    motor_input_source="model",
    setpoint_prediction_state="prediction",
):
    n_starts = len(start_indices)
    progress(f"build_multistep_predictions start: {n_starts} starts x {horizon} steps")
    prediction_progress_stride = max(1, min(10, n_starts // 10 if n_starts >= 10 else 1))
    pred = np.full((n_starts, horizon + 1, len(STATE_NAMES)), np.nan, dtype=float)
    pred_calc = {
        "vel_sp": np.full((n_starts, horizon, 3), np.nan),
        "acc_sp": np.full((n_starts, horizon, 3), np.nan),
        "att_sp": np.full((n_starts, horizon, 4), np.nan),
        "rate_sp": np.full((n_starts, horizon, 3), np.nan),
        "thr_sp": np.full((n_starts, horizon, 3), np.nan),
        "torque_sp": np.full((n_starts, horizon, 3), np.nan),
        "torque_sp_unfiltered": np.full((n_starts, horizon, 3), np.nan),
        "torque_p": np.full((n_starts, horizon, 3), np.nan),
        "torque_i": np.full((n_starts, horizon, 3), np.nan),
        "torque_d": np.full((n_starts, horizon, 3), np.nan),
        "torque_ff": np.full((n_starts, horizon, 3), np.nan),
        "motor_sp": np.full((n_starts, horizon, 4), np.nan),
        "motor_from_px4_torque": np.full((n_starts, horizon, 4), np.nan),
        "phys_acc": np.full((n_starts, horizon, 3), np.nan),
        "thrust_acc": np.full((n_starts, horizon, 3), np.nan),
        "acc_used": np.full((n_starts, horizon, 3), np.nan),
        "actuator_acc": np.full((n_starts, horizon, 3), np.nan),
        "acc_residual": np.full((n_starts, horizon, 3), np.nan),
        "phys_wdot": np.full((n_starts, horizon, 3), np.nan),
        "rate_derivative": np.full((n_starts, horizon, 3), np.nan),
        "rate_int": np.full((n_starts, horizon, 3), np.nan),
        "allocator_unallocated_torque": np.full((n_starts, horizon, 3), np.nan),
        "phys_force": np.full((n_starts, horizon, 3), np.nan),
        "phys_torque": np.full((n_starts, horizon, 3), np.nan),
        "phys_torque_raw": np.full((n_starts, horizon, 3), np.nan),
        "bias_torque": np.full((n_starts, horizon, 3), np.nan),
        "motor_speed": np.full((n_starts, horizon, 4), np.nan),
        "acceleration_bias": np.full((n_starts, horizon, 3), np.nan),
        "z_velocity_bias": np.full((n_starts, horizon, 1), np.nan),
        "imu_acc": np.full((n_starts, horizon, 3), np.nan),
    }

    for out_i, start in enumerate(start_indices):
        if out_i == 0 or (out_i + 1) % prediction_progress_stride == 0 or out_i + 1 == n_starts:
            start_time = float(df.iloc[start]["control_time_s"])
            progress(
                f"build_multistep_predictions {out_i + 1}/{n_starts} "
                f"({100.0 * (out_i + 1) / max(n_starts, 1):.0f}%, start_t={start_time:.3f}s)"
            )
        s = actual[start].copy()
        if not np.all(np.isfinite(s)):
            continue
        pred[out_i, 0] = s.copy()

        prev_vel = finite_vector(context_hist["prev_vel"][start], s[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]])
        prev_acc = finite_vector(context_hist["prev_acc"][start], np.zeros(3))
        vel_int = finite_vector(context_hist["vel_int"][start], np.zeros(3))
        logged_vel_int = estimate_vel_int_from_logged_setpoint(df.iloc[start], s, prev_acc)
        if logged_vel_int is not None:
            vel_int = logged_vel_int
        prev_omega = finite_vector(context_hist["prev_omega"][start], s[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]])
        prev_omega_dot = finite_vector(context_hist["prev_omega_dot"][start], np.zeros(3))
        rate_int = finite_vector(context_hist["rate_int"][start], np.zeros(3))
        motor_speed = finite_vector(context_hist["motor_speed"][start], np.zeros(4))
        yaw_torque_lpf_state = context_hist["yaw_torque_lpf_state"][start]
        if not np.isfinite(yaw_torque_lpf_state):
            yaw_torque_lpf_state = None
        allocator_saturation_positive = context_hist["allocator_saturation_positive"][start].copy()
        allocator_saturation_negative = context_hist["allocator_saturation_negative"][start].copy()
        rate_delay_buffer = [entry.copy() for entry in context_hist["rate_delay_buffer"][start]]
        motor_command_delay_buffer = [entry.copy() for entry in context_hist["motor_command_delay_buffer"][start]]
        prediction_z_vel_max_up = finite_scalar(context_hist["z_vel_max_up"][start], MPC_Z_VEL_MAX_UP)
        prediction_thrust_min = finite_scalar(context_hist["thrust_min"][start], MPC_THR_MIN)
        prediction_no_thrust = bool(context_hist["no_thrust"][start])
        prediction_hover_thrust = finite_scalar(context_hist["hover_thrust"][start], MPC_THR_HOVER)
        prediction_tilt_limit = finite_scalar(context_hist["tilt_limit"][start], MPC_TILT_MAX)
        prediction_landed_or_maybe_landed = bool(context_hist["landed_or_maybe_landed"][start])
        prediction_acceleration_bias = finite_vector(context_hist["acceleration_bias"][start], np.zeros(3))
        prediction_z_velocity_bias = finite_scalar(context_hist["z_velocity_bias"][start], 0.0)
        prediction_acceleration_bias_rate = np.zeros(3)
        if USE_ACCELERATION_BIAS_TREND_PREDICTION and np.any(VELOCITY_DELTA_BIAS_AXES):
            lookback = int(max(1, ACCELERATION_BIAS_TREND_LOOKBACK_STEPS))
            prev_start = max(0, start - lookback)
            if "_segment_id" in df.columns:
                while prev_start < start and df.iloc[prev_start]["_segment_id"] != df.iloc[start]["_segment_id"]:
                    prev_start += 1
            if prev_start < start:
                prev_bias = finite_vector(
                    context_hist["acceleration_bias"][prev_start],
                    prediction_acceleration_bias,
                )
                elapsed = float(df.iloc[start]["control_time_s"] - df.iloc[prev_start]["control_time_s"])
                if np.isfinite(elapsed) and elapsed > 1.0e-6:
                    prediction_acceleration_bias_rate = (
                        prediction_acceleration_bias - prev_bias
                    ) / elapsed
                    prediction_acceleration_bias_rate = np.clip(
                        prediction_acceleration_bias_rate,
                        -ACCELERATION_BIAS_TREND_RATE_LIMIT,
                        ACCELERATION_BIAS_TREND_RATE_LIMIT,
                    )
                    prediction_acceleration_bias_rate[~VELOCITY_DELTA_BIAS_AXES] = 0.0
        plot_prev_vel = prev_vel.copy()
        plot_prev_acc = prev_acc.copy()
        plot_vel_int = vel_int.copy()
        plot_prev_omega = prev_omega.copy()
        plot_prev_omega_dot = prev_omega_dot.copy()
        plot_rate_int = rate_int.copy()
        plot_motor_speed = motor_speed.copy()
        plot_yaw_torque_lpf_state = yaw_torque_lpf_state
        plot_allocator_saturation_positive = allocator_saturation_positive.copy()
        plot_allocator_saturation_negative = allocator_saturation_negative.copy()
        plot_rate_delay_buffer = [entry.copy() for entry in rate_delay_buffer]
        plot_motor_command_delay_buffer = [entry.copy() for entry in motor_command_delay_buffer]

        for h in range(horizon):
            row_idx = start + h
            if row_idx >= len(df):
                break

            input_row = df.iloc[row_idx]
            pos_sp, yaw_sp = input_from_row(input_row, use_local_sp=use_local_sp_position)
            step_acceleration_bias = prediction_acceleration_bias + prediction_acceleration_bias_rate * (h * step_dt)
            step_acceleration_bias = np.clip(
                step_acceleration_bias,
                -ACCELERATION_BIAS_LIMIT,
                ACCELERATION_BIAS_LIMIT,
            )
            s_current = s.copy()
            s, prev_acc, vel_int, debug = cu_mpc_simulator_step(
                s,
                pos_sp,
                yaw_sp,
                prev_vel,
                prev_acc,
                vel_int,
                step_dt,
                prev_omega=prev_omega,
                rate_int=rate_int,
                z_vel_max_up=prediction_z_vel_max_up,
                thrust_min=prediction_thrust_min,
                no_thrust=prediction_no_thrust,
                hover_thrust=prediction_hover_thrust,
                tilt_limit=prediction_tilt_limit,
                rate_delay_buffer=rate_delay_buffer,
                rate_delay_steps=rate_delay_steps,
                motor_command_delay_buffer=motor_command_delay_buffer,
                motor_command_delay_steps=motor_command_delay_steps,
                landed_or_maybe_landed=prediction_landed_or_maybe_landed,
                saturation_positive=allocator_saturation_positive,
                saturation_negative=allocator_saturation_negative,
                yaw_torque_lpf_state=yaw_torque_lpf_state,
                prev_motor_speed=motor_speed,
                prev_omega_dot=prev_omega_dot,
                acceleration_bias=step_acceleration_bias,
                z_position_velocity_bias=(
                    prediction_z_velocity_bias
                    if USE_Z_POSITION_VELOCITY_BIAS_CORRECTION
                    else 0.0
                ),
                actuator_motor_override=(
                    actuator_motor_from_row(input_row)
                    if motor_input_source == "actuator_log"
                    else None
                ),
            )
            debug_for_plot = debug
            if setpoint_prediction_state == "actual_log":
                actual_state = actual[row_idx]
                if np.all(np.isfinite(actual_state)):
                    _, plot_prev_acc_next, plot_vel_int_next, debug_for_plot = cu_mpc_simulator_step(
                        actual_state.copy(),
                        pos_sp,
                        yaw_sp,
                        plot_prev_vel,
                        plot_prev_acc,
                        plot_vel_int,
                        step_dt,
                        prev_omega=plot_prev_omega,
                        rate_int=plot_rate_int,
                        z_vel_max_up=prediction_z_vel_max_up,
                        thrust_min=prediction_thrust_min,
                        no_thrust=prediction_no_thrust,
                        hover_thrust=prediction_hover_thrust,
                        tilt_limit=prediction_tilt_limit,
                        rate_delay_buffer=plot_rate_delay_buffer,
                        rate_delay_steps=rate_delay_steps,
                        motor_command_delay_buffer=plot_motor_command_delay_buffer,
                        motor_command_delay_steps=motor_command_delay_steps,
                        landed_or_maybe_landed=prediction_landed_or_maybe_landed,
                        saturation_positive=plot_allocator_saturation_positive,
                        saturation_negative=plot_allocator_saturation_negative,
                        yaw_torque_lpf_state=plot_yaw_torque_lpf_state,
                        prev_motor_speed=plot_motor_speed,
                        prev_omega_dot=plot_prev_omega_dot,
                        acceleration_bias=step_acceleration_bias,
                        z_position_velocity_bias=(
                            prediction_z_velocity_bias
                            if USE_Z_POSITION_VELOCITY_BIAS_CORRECTION
                            else 0.0
                        ),
                        actuator_motor_override=(
                            actuator_motor_from_row(input_row)
                            if motor_input_source == "actuator_log"
                            else None
                        ),
                    )
                    plot_rate_int = debug_for_plot.get("rate_int_next", plot_rate_int)
                    plot_prev_omega_dot = debug_for_plot.get("rate_derivative_next", plot_prev_omega_dot)
                    plot_motor_speed = debug_for_plot.get("motor_speed", plot_motor_speed)
                    plot_yaw_torque_lpf_state = debug_for_plot.get("yaw_torque_lpf_next", plot_yaw_torque_lpf_state)
                    plot_allocator_saturation_positive = debug_for_plot.get(
                        "allocator_saturation_positive",
                        plot_allocator_saturation_positive,
                    )
                    plot_allocator_saturation_negative = debug_for_plot.get(
                        "allocator_saturation_negative",
                        plot_allocator_saturation_negative,
                    )
                    plot_prev_vel = actual_state[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]].copy()
                    plot_prev_omega = actual_state[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]].copy()
                    plot_prev_acc = plot_prev_acc_next
                    plot_vel_int = plot_vel_int_next
            pred[out_i, h + 1] = s.copy()
            for key in pred_calc:
                if key in debug_for_plot:
                    pred_calc[key][out_i, h] = debug_for_plot[key]
            pred_calc["acceleration_bias"][out_i, h] = step_acceleration_bias
            pred_calc["z_velocity_bias"][out_i, h, 0] = (
                prediction_z_velocity_bias
                if USE_Z_POSITION_VELOCITY_BIAS_CORRECTION
                else 0.0
            )

            rate_int = debug.get("rate_int_next", rate_int)
            prev_omega_dot = debug.get("rate_derivative_next", prev_omega_dot)
            motor_speed = debug.get("motor_speed", motor_speed)
            yaw_torque_lpf_state = debug.get("yaw_torque_lpf_next", yaw_torque_lpf_state)
            allocator_saturation_positive = debug.get("allocator_saturation_positive", allocator_saturation_positive)
            allocator_saturation_negative = debug.get("allocator_saturation_negative", allocator_saturation_negative)
            prev_vel = s_current[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]].copy()
            prev_omega = s_current[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]].copy()

    progress("build_multistep_predictions done")
    return pred, pred_calc


def add_segments(ax, segments, color, label, linewidth=1.5, alpha=0.8, linestyle="-"):
    if not segments:
        return
    collection = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        linestyles=linestyle,
    )
    ax.add_collection(collection)
    ax.plot([], [], color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label)


def state_prediction_segments(plot_t, start_indices, pred_rollouts, actual, value_func, step_dt):
    pred_segments = []
    error_segments = []
    for start, pred in zip(start_indices, pred_rollouts):
        pred_value = value_func(pred)
        pred_mask = np.isfinite(pred_value)
        if np.any(pred_mask):
            length = int(np.flatnonzero(pred_mask)[-1]) + 1
            xs = plot_t[start] + np.arange(length) * step_dt
            pred_segments.append(np.column_stack([xs, pred_value[:length]]))

        actual_len = min(len(pred), len(actual) - start)
        if actual_len <= 0:
            continue
        actual_value = value_func(actual[start:start + actual_len])
        error_value = pred_value[:actual_len] - actual_value
        error_mask = np.isfinite(error_value)
        if np.any(error_mask):
            length = int(np.flatnonzero(error_mask)[-1]) + 1
            xs = plot_t[start] + np.arange(length) * step_dt
            error_segments.append(np.column_stack([xs, error_value[:length]]))
    return pred_segments, error_segments


def state_delta_segments(plot_t, start_indices, pred_rollouts, actual, value_func, step_dt):
    pred_segments = []
    actual_segments = []
    for start, pred in zip(start_indices, pred_rollouts):
        pred_value = value_func(pred)
        if len(pred_value) == 0 or not np.isfinite(pred_value[0]):
            continue
        pred_delta = pred_value - pred_value[0]
        pred_mask = np.isfinite(pred_delta)
        if np.any(pred_mask):
            length = int(np.flatnonzero(pred_mask)[-1]) + 1
            xs = plot_t[start] + np.arange(length) * step_dt
            pred_segments.append(np.column_stack([xs, pred_delta[:length]]))

        actual_len = min(len(pred), len(actual) - start)
        if actual_len <= 0:
            continue
        actual_value = value_func(actual[start:start + actual_len])
        if not np.isfinite(actual_value[0]):
            continue
        actual_delta = actual_value - actual_value[0]
        actual_mask = np.isfinite(actual_delta)
        if np.any(actual_mask):
            length = int(np.flatnonzero(actual_mask)[-1]) + 1
            xs = plot_t[start] + np.arange(length) * step_dt
            actual_segments.append(np.column_stack([xs, actual_delta[:length]]))
    return pred_segments, actual_segments


def setpoint_prediction_segments(plot_t, start_indices, pred_calc, key, axis, step_dt):
    segments = []
    if pred_calc is None or key not in pred_calc:
        return segments
    for start, values in zip(start_indices, pred_calc[key]):
        pred_value = values[:, axis]
        pred_mask = np.isfinite(pred_value)
        if np.any(pred_mask):
            length = int(np.flatnonzero(pred_mask)[-1]) + 1
            xs = plot_t[start] + np.arange(length + 1) * step_dt
            y = np.r_[pred_value[:length], pred_value[length - 1]]
            segments.append(np.column_stack([xs, y]))
    return segments


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


def plot_window_mask(plot_t, start_time, end_time):
    mask = np.isfinite(plot_t)
    mask &= plot_t + 1.0e-9 >= start_time
    mask &= plot_t <= end_time + 1.0e-9
    return mask


def timestamp_column_for_log_column(log_col):
    if log_col is None:
        return None
    if log_col.startswith("local_sp_"):
        return "local_sp_timestamp"
    if log_col.startswith("att_sp_"):
        return "att_sp_timestamp"
    if log_col.startswith("rate_sp_"):
        return "rate_sp_timestamp"
    if log_col.startswith("thrust_sp_"):
        return "thrust_sp_timestamp"
    if log_col.startswith("torque_sp_"):
        return "torque_sp_timestamp"
    if log_col.startswith("actuator_motor_"):
        return "actuator_timestamp"
    if log_col in RATE_INT_COLUMNS:
        return "rate_ctrl_status_timestamp"
    return None


def plot_time_for_log_column(df, log_col, fallback_plot_t, base_timestamp):
    timestamp_col = timestamp_column_for_log_column(log_col)
    if timestamp_col is None or timestamp_col not in df.columns or not np.isfinite(base_timestamp):
        return fallback_plot_t

    timestamp = df[timestamp_col].to_numpy(float)
    log_plot_t = (timestamp - base_timestamp) * 1.0e-6
    valid = np.isfinite(timestamp) & (timestamp > 1.0e12)
    return np.where(valid, log_plot_t, fallback_plot_t)


def setpoint_legend_base(key):
    bases = {
        "vel_sp": "velocity_setpoint",
        "acc_sp": "acceleration_setpoint",
        "att_sp": "attitude_setpoint",
        "rate_sp": "omega_setpoint",
        "thr_sp": "thrust_setpoint",
        "torque_sp": "torque_setpoint",
        "torque_sp_unfiltered": "torque_setpoint_unfiltered",
        "torque_p": "torque_p",
        "torque_i": "torque_i",
        "torque_d": "torque_d",
        "torque_ff": "torque_ff",
        "motor_sp": "actuator_motor",
        "motor_from_px4_torque": "allocator",
        "phys_acc": "physical_acceleration",
        "thrust_acc": "thrust_attitude_acceleration",
        "acc_used": "delayed_acceleration_used",
        "actuator_acc": "logged_actuator_acceleration",
        "acc_residual": "imu_minus_model_acceleration",
        "phys_wdot": "physical_angular_acceleration",
        "rate_derivative": "rate_derivative",
        "rate_int": "rate_integral",
        "allocator_unallocated_torque": "allocator_unallocated_torque",
        "phys_force": "physical_force_body",
        "phys_torque": "physical_torque_body",
        "phys_torque_raw": "physical_torque_body_raw",
        "bias_torque": "rate_int_bias_torque",
        "motor_speed": "motor_speed",
        "acceleration_bias": "acceleration_bias",
        "z_velocity_bias": "z_velocity_bias",
        "imu_acc": "livox_imu_acceleration",
    }
    return bases.get(key, key)


def setpoint_component_name(key, axis):
    components = {
        "vel_sp": ["velocity_x_setpoint", "velocity_y_setpoint", "velocity_z_setpoint"],
        "acc_sp": ["acceleration_x_setpoint", "acceleration_y_setpoint", "acceleration_z_setpoint"],
        "att_sp": ["attitude_qw_setpoint", "attitude_qx_setpoint", "attitude_qy_setpoint", "attitude_qz_setpoint"],
        "rate_sp": ["roll_rate_setpoint", "pitch_rate_setpoint", "yaw_rate_setpoint"],
        "thr_sp": ["thrust_x_setpoint", "thrust_y_setpoint", "thrust_z_setpoint"],
        "torque_sp": ["torque_x_setpoint", "torque_y_setpoint", "torque_z_setpoint"],
        "torque_sp_unfiltered": ["torque_x_unfiltered", "torque_y_unfiltered", "torque_z_unfiltered"],
        "torque_p": ["torque_p_x", "torque_p_y", "torque_p_z"],
        "torque_i": ["torque_i_x", "torque_i_y", "torque_i_z"],
        "torque_d": ["torque_d_x", "torque_d_y", "torque_d_z"],
        "torque_ff": ["torque_ff_x", "torque_ff_y", "torque_ff_z"],
        "motor_sp": ["actuator_motor_0", "actuator_motor_1", "actuator_motor_2", "actuator_motor_3"],
        "motor_from_px4_torque": [
            "allocator_0",
            "allocator_1",
            "allocator_2",
            "allocator_3",
        ],
        "phys_acc": ["physical_acceleration_x", "physical_acceleration_y", "physical_acceleration_z"],
        "thrust_acc": ["thrust_attitude_acceleration_x", "thrust_attitude_acceleration_y", "thrust_attitude_acceleration_z"],
        "acc_used": ["delayed_acceleration_used_x", "delayed_acceleration_used_y", "delayed_acceleration_used_z"],
        "actuator_acc": ["logged_actuator_acceleration_x", "logged_actuator_acceleration_y", "logged_actuator_acceleration_z"],
        "acc_residual": ["imu_minus_model_acceleration_x", "imu_minus_model_acceleration_y", "imu_minus_model_acceleration_z"],
        "phys_wdot": ["physical_angular_acceleration_x", "physical_angular_acceleration_y", "physical_angular_acceleration_z"],
        "rate_derivative": ["rate_derivative_x", "rate_derivative_y", "rate_derivative_z"],
        "rate_int": ["rate_integral_x", "rate_integral_y", "rate_integral_z"],
        "allocator_unallocated_torque": ["unallocated_torque_x", "unallocated_torque_y", "unallocated_torque_z"],
        "phys_force": ["physical_force_body_x", "physical_force_body_y", "physical_force_body_z"],
        "phys_torque": ["physical_torque_body_x", "physical_torque_body_y", "physical_torque_body_z"],
        "phys_torque_raw": ["physical_torque_body_raw_x", "physical_torque_body_raw_y", "physical_torque_body_raw_z"],
        "bias_torque": ["rate_int_bias_torque_x", "rate_int_bias_torque_y", "rate_int_bias_torque_z"],
        "motor_speed": ["motor_speed_0", "motor_speed_1", "motor_speed_2", "motor_speed_3"],
        "acceleration_bias": ["acceleration_bias_x", "acceleration_bias_y", "acceleration_bias_z"],
        "z_velocity_bias": ["z_velocity_bias"],
        "imu_acc": ["livox_imu_acceleration_x", "livox_imu_acceleration_y", "livox_imu_acceleration_z"],
    }
    names = components.get(key)
    if names is None or axis >= len(names):
        return setpoint_legend_base(key)
    return names[axis]


def first_existing_column(df, columns):
    if columns is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        if column in df.columns:
            return column
    return None


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
    global USE_RATE_INT_BIAS_TORQUE_WITH_ACTUATOR_LOG
    parser = argparse.ArgumentParser(description="Compare offboard log setpoints and recursive dynamics predictions against mpc_simulator.cu-style prediction.")
    parser.add_argument("state", nargs="?", default="wx")
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--rate-delay-steps", type=int, default=RATE_DELAY_STEPS)
    parser.add_argument("--motor-command-delay-steps", type=int, default=MOTOR_COMMAND_DELAY_STEPS)
    parser.add_argument(
        "--actuator-log-shift-steps",
        type=int,
        default=ACTUATOR_LOG_SHIFT_STEPS,
        help="sample shift applied to actuator_motor_* for actuator_log prediction; negative uses newer log samples.",
    )
    parser.add_argument(
        "--rate-int-bias-with-actuator-log",
        action=argparse.BooleanOptionalAction,
        default=USE_RATE_INT_BIAS_TORQUE_WITH_ACTUATOR_LOG,
        help="apply the same rate-integrator bias torque correction when actuator_log drives the physical model.",
    )
    parser.add_argument("--rate-int-sync-period", type=float, default=RATE_INT_SYNC_PERIOD)
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON, help="recursive prediction horizon in steps (1..75)")
    parser.add_argument("--prediction-interval", type=float, default=PREDICTION_INTERVAL, help="time spacing between recursive prediction starts [s]")
    parser.add_argument("--plot-start", type=float, default=0.0, help="display start time on the plotted time axis [s]")
    parser.add_argument("--plot-end", type=float, default=None, help="display end time on the plotted time axis [s]")
    parser.add_argument(
        "--input-position-source",
        choices=("target", "local_sp"),
        default="target",
        help="position setpoint source for model input. target uses delayed target_*; local_sp uses local_sp_x/y/z only.",
    )
    parser.add_argument(
        "--motor-input-source",
        choices=("model", "actuator_log"),
        default="actuator_log",
        help="motor input used for physical update. model uses allocated motor_setpoint; actuator_log uses logged actuator_motor_0..3.",
    )
    parser.add_argument(
        "--setpoint-prediction-state",
        choices=("prediction", "actual_log"),
        default="prediction",
        help="state used to recompute setpoint prediction lines. actual_log is diagnostic and uses future logged states.",
    )
    args = parser.parse_args()
    USE_RATE_INT_BIAS_TORQUE_WITH_ACTUATOR_LOG = args.rate_int_bias_with_actuator_log
    prediction_horizon = int(np.clip(args.horizon, 1, PREDICTION_HORIZON))
    if prediction_horizon != args.horizon:
        print(f"[INFO] clipped horizon from {args.horizon} to {prediction_horizon}")
    plot_start = max(0.0, float(args.plot_start))
    default_plot_end = plot_start + prediction_horizon * args.dt
    plot_end = default_plot_end if args.plot_end is None else float(args.plot_end)
    if plot_end < plot_start:
        print(f"[ERROR] --plot-end ({plot_end:.3f}) must be >= --plot-start ({plot_start:.3f})")
        sys.exit(1)

    progress(f"read csv: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    progress(f"read csv done: {len(df)} raw rows")
    progress("preprocess log")
    df = preprocess_log(df, args.dt)
    df = add_delayed_target_columns(df, TARGET_LOCAL_SP_DELAY_STEPS)
    df = add_actuator_input_columns(df, args.actuator_log_shift_steps)
    progress(f"preprocess done: {len(df)} samples after dt binning")
    t = df["control_time_s"].to_numpy(float)
    t_diff = np.diff(t)
    finite_dt = t_diff[np.isfinite(t_diff) & (t_diff > 0.0)]
    csv_dt = float(np.median(finite_dt)) if len(finite_dt) > 0 else args.dt

    history = build_history(
        df,
        args.dt,
        rate_delay_steps=args.rate_delay_steps,
        motor_command_delay_steps=args.motor_command_delay_steps,
        rate_int_sync_period=args.rate_int_sync_period,
        return_context=prediction_horizon > 1,
        use_local_sp_position=args.input_position_source == "local_sp",
        motor_input_source=args.motor_input_source,
    )
    if prediction_horizon > 1:
        actual, actual_next, pred_next, calc, context_hist = history
    else:
        actual, actual_next, pred_next, calc = history
        context_hist = None
    base_active_plot_mask = non_idle_plot_mask(df)
    if np.any(base_active_plot_mask):
        first_active_idx = int(np.flatnonzero(base_active_plot_mask)[0])
        plot_t = t - t[first_active_idx]
    else:
        first_active_idx = 0
        plot_t = t.copy()
    base_timestamp = (
        float(df.iloc[first_active_idx]["sync_timestamp"])
        if "sync_timestamp" in df.columns and len(df) > first_active_idx
        else np.nan
    )
    active_plot_mask = base_active_plot_mask & plot_window_mask(plot_t, plot_start, plot_end)
    if not np.any(active_plot_mask):
        print(f"[WARN] no samples in plot window {plot_start:.3f}..{plot_end:.3f}s")
    prediction_start_indices = np.array([], dtype=int)
    pred_rollouts = None
    pred_calc_rollouts = None
    if prediction_horizon > 1:
        prediction_duration = prediction_horizon * args.dt
        prediction_start_end = plot_end - prediction_duration
        prediction_start_mask = base_active_plot_mask & plot_window_mask(
            plot_t,
            plot_start,
            prediction_start_end,
        )
        prediction_start_indices = selected_prediction_start_indices(
            plot_t,
            prediction_start_mask,
            args.prediction_interval,
        )
        progress(f"selected {len(prediction_start_indices)} prediction starts")
        pred_rollouts, pred_calc_rollouts = build_multistep_predictions(
            df,
            actual,
            context_hist,
            prediction_start_indices,
            args.dt,
            prediction_horizon,
            rate_delay_steps=args.rate_delay_steps,
            motor_command_delay_steps=args.motor_command_delay_steps,
            use_local_sp_position=args.input_position_source == "local_sp",
            motor_input_source=args.motor_input_source,
            setpoint_prediction_state=args.setpoint_prediction_state,
        )

    progress("prepare plot data")
    state_aliases = {
        "e0": "q0", "e1": "q1", "e2": "q2", "e3": "q3",
        "pos_x": "x", "pos_y": "y", "pos_z": "z",
        "vel_x": "vx", "vel_y": "vy", "vel_z": "vz",
        "local_pos_z_deriv": "z_deriv",
        "angular_vel_x": "wx", "angular_vel_y": "wy", "angular_vel_z": "wz",
        "roll_rate": "wx", "rolll_rate": "wx", "pitch_rate": "wy", "yaw_rate": "wz",
        "local_sp_vx": "vel_sp_x", "local_sp_vy": "vel_sp_y", "local_sp_vz": "vel_sp_z",
        "local_sp_ax": "acc_sp_x", "local_sp_ay": "acc_sp_y", "local_sp_az": "acc_sp_z",
        "thrust_x": "thr_sp_x", "thrust_y": "thr_sp_y", "thrust_z": "thr_sp_z",
        "torque_x": "torque_sp_x", "torque_y": "torque_sp_y", "torque_z": "torque_sp_z",
        "motor0": "actuator_motor_0", "motor1": "actuator_motor_1",
        "motor2": "actuator_motor_2", "motor3": "actuator_motor_3",
        "allocator0": "allocator_0", "allocator1": "allocator_1",
        "allocator2": "allocator_2", "allocator3": "allocator_3",
    }
    requested_state = args.state.strip()
    delta_mode = requested_state.startswith("d") and len(requested_state) > 1
    state_name = requested_state[1:] if delta_mode else requested_state
    state = state_aliases.get(state_name, state_name)

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
        "thr_sp_x": ("thr_sp", 0, ["thrust_sp_x", "rate_sp_thrust_x"]),
        "thr_sp_y": ("thr_sp", 1, ["thrust_sp_y", "rate_sp_thrust_y"]),
        "thr_sp_z": ("thr_sp", 2, ["thrust_sp_z", "rate_sp_thrust_z"]),
        "torque_sp_x": ("torque_sp", 0, "torque_sp_x"),
        "torque_sp_y": ("torque_sp", 1, "torque_sp_y"),
        "torque_sp_z": ("torque_sp", 2, "torque_sp_z"),
        "torque_unfiltered_x": ("torque_sp_unfiltered", 0, "torque_sp_x"),
        "torque_unfiltered_y": ("torque_sp_unfiltered", 1, "torque_sp_y"),
        "torque_unfiltered_z": ("torque_sp_unfiltered", 2, "torque_sp_z"),
        "torque_p_x": ("torque_p", 0, None),
        "torque_p_y": ("torque_p", 1, None),
        "torque_p_z": ("torque_p", 2, None),
        "torque_i_x": ("torque_i", 0, None),
        "torque_i_y": ("torque_i", 1, None),
        "torque_i_z": ("torque_i", 2, None),
        "torque_d_x": ("torque_d", 0, None),
        "torque_d_y": ("torque_d", 1, None),
        "torque_d_z": ("torque_d", 2, None),
        "torque_ff_x": ("torque_ff", 0, None),
        "torque_ff_y": ("torque_ff", 1, None),
        "torque_ff_z": ("torque_ff", 2, None),
        "actuator_motor_0": ("motor_sp", 0, "actuator_motor_0"),
        "actuator_motor_1": ("motor_sp", 1, "actuator_motor_1"),
        "actuator_motor_2": ("motor_sp", 2, "actuator_motor_2"),
        "actuator_motor_3": ("motor_sp", 3, "actuator_motor_3"),
        "actuator_motor_0_px4mix": ("motor_from_px4_torque", 0, "actuator_motor_0"),
        "actuator_motor_1_px4mix": ("motor_from_px4_torque", 1, "actuator_motor_1"),
        "actuator_motor_2_px4mix": ("motor_from_px4_torque", 2, "actuator_motor_2"),
        "actuator_motor_3_px4mix": ("motor_from_px4_torque", 3, "actuator_motor_3"),
        "allocator_0": ("motor_from_px4_torque", 0, "actuator_motor_0"),
        "allocator_1": ("motor_from_px4_torque", 1, "actuator_motor_1"),
        "allocator_2": ("motor_from_px4_torque", 2, "actuator_motor_2"),
        "allocator_3": ("motor_from_px4_torque", 3, "actuator_motor_3"),
        "motor_speed_0": ("motor_speed", 0, None),
        "motor_speed_1": ("motor_speed", 1, None),
        "motor_speed_2": ("motor_speed", 2, None),
        "motor_speed_3": ("motor_speed", 3, None),
        "acceleration_bias_x": ("acceleration_bias", 0, None),
        "acceleration_bias_y": ("acceleration_bias", 1, None),
        "acceleration_bias_z": ("acceleration_bias", 2, None),
        "z_velocity_bias": ("z_velocity_bias", 0, None),
        "imu_acc_x": ("imu_acc", 0, "livox_imu_acc_x"),
        "imu_acc_y": ("imu_acc", 1, "livox_imu_acc_y"),
        "imu_acc_z": ("imu_acc", 2, "livox_imu_acc_z"),
        "livox_imu_acc_x": ("imu_acc", 0, "livox_imu_acc_x"),
        "livox_imu_acc_y": ("imu_acc", 1, "livox_imu_acc_y"),
        "livox_imu_acc_z": ("imu_acc", 2, "livox_imu_acc_z"),
        "phys_acc_x": ("phys_acc", 0, None),
        "phys_acc_y": ("phys_acc", 1, None),
        "phys_acc_z": ("phys_acc", 2, None),
        "thrust_acc_x": ("thrust_acc", 0, None),
        "thrust_acc_y": ("thrust_acc", 1, None),
        "thrust_acc_z": ("thrust_acc", 2, None),
        "acc_used_x": ("acc_used", 0, None),
        "acc_used_y": ("acc_used", 1, None),
        "acc_used_z": ("acc_used", 2, None),
        "actuator_acc_x": ("actuator_acc", 0, None),
        "actuator_acc_y": ("actuator_acc", 1, None),
        "actuator_acc_z": ("actuator_acc", 2, None),
        "acc_residual_x": ("acc_residual", 0, None),
        "acc_residual_y": ("acc_residual", 1, None),
        "acc_residual_z": ("acc_residual", 2, None),
        "phys_wdot_x": ("phys_wdot", 0, None),
        "phys_wdot_y": ("phys_wdot", 1, None),
        "phys_wdot_z": ("phys_wdot", 2, None),
        "rate_derivative_x": ("rate_derivative", 0, None),
        "rate_derivative_y": ("rate_derivative", 1, None),
        "rate_derivative_z": ("rate_derivative", 2, None),
        "rate_int_x": ("rate_int", 0, None),
        "rate_int_y": ("rate_int", 1, None),
        "rate_int_z": ("rate_int", 2, None),
        "unallocated_torque_x": ("allocator_unallocated_torque", 0, None),
        "unallocated_torque_y": ("allocator_unallocated_torque", 1, None),
        "unallocated_torque_z": ("allocator_unallocated_torque", 2, None),
        "phys_force_x": ("phys_force", 0, None),
        "phys_force_y": ("phys_force", 1, None),
        "phys_force_z": ("phys_force", 2, None),
        "phys_torque_x": ("phys_torque", 0, None),
        "phys_torque_y": ("phys_torque", 1, None),
        "phys_torque_z": ("phys_torque", 2, None),
        "phys_torque_raw_x": ("phys_torque_raw", 0, None),
        "phys_torque_raw_y": ("phys_torque_raw", 1, None),
        "phys_torque_raw_z": ("phys_torque_raw", 2, None),
        "bias_torque_x": ("bias_torque", 0, None),
        "bias_torque_y": ("bias_torque", 1, None),
        "bias_torque_z": ("bias_torque", 2, None),
    }
    setpoint_groups = {
        "thrust": ["thr_sp_x", "thr_sp_y", "thr_sp_z"],
        "thr_sp": ["thr_sp_x", "thr_sp_y", "thr_sp_z"],
        "torque": ["torque_sp_x", "torque_sp_y", "torque_sp_z"],
        "torque_sp": ["torque_sp_x", "torque_sp_y", "torque_sp_z"],
        "torque_unfiltered": ["torque_unfiltered_x", "torque_unfiltered_y", "torque_unfiltered_z"],
        "torque_p": ["torque_p_x", "torque_p_y", "torque_p_z"],
        "torque_i": ["torque_i_x", "torque_i_y", "torque_i_z"],
        "torque_d": ["torque_d_x", "torque_d_y", "torque_d_z"],
        "torque_ff": ["torque_ff_x", "torque_ff_y", "torque_ff_z"],
        "actuator_motors": ["actuator_motor_0", "actuator_motor_1", "actuator_motor_2", "actuator_motor_3"],
        "motors": ["actuator_motor_0", "actuator_motor_1", "actuator_motor_2", "actuator_motor_3"],
        "actuator_motors_px4mix": [
            "actuator_motor_0_px4mix",
            "actuator_motor_1_px4mix",
            "actuator_motor_2_px4mix",
            "actuator_motor_3_px4mix",
        ],
        "allocator": ["allocator_0", "allocator_1", "allocator_2", "allocator_3"],
        "phys_acc": ["phys_acc_x", "phys_acc_y", "phys_acc_z"],
        "thrust_acc": ["thrust_acc_x", "thrust_acc_y", "thrust_acc_z"],
        "acc_used": ["acc_used_x", "acc_used_y", "acc_used_z"],
        "actuator_acc": ["actuator_acc_x", "actuator_acc_y", "actuator_acc_z"],
        "acc_residual": ["acc_residual_x", "acc_residual_y", "acc_residual_z"],
        "acc_y_diag": [
            "acc_sp_y",
            "thrust_acc_y",
            "phys_acc_y",
            "acc_used_y",
            "actuator_acc_y",
            "imu_acc_y",
            "acc_residual_y",
        ],
        "phys_wdot": ["phys_wdot_x", "phys_wdot_y", "phys_wdot_z"],
        "rate_derivative": ["rate_derivative_x", "rate_derivative_y", "rate_derivative_z"],
        "rate_int": ["rate_int_x", "rate_int_y", "rate_int_z"],
        "unallocated_torque": ["unallocated_torque_x", "unallocated_torque_y", "unallocated_torque_z"],
        "phys_force": ["phys_force_x", "phys_force_y", "phys_force_z"],
        "phys_torque": ["phys_torque_x", "phys_torque_y", "phys_torque_z"],
        "phys_torque_raw": ["phys_torque_raw_x", "phys_torque_raw_y", "phys_torque_raw_z"],
        "bias_torque": ["bias_torque_x", "bias_torque_y", "bias_torque_z"],
        "motor_speed": ["motor_speed_0", "motor_speed_1", "motor_speed_2", "motor_speed_3"],
        "acceleration_bias": ["acceleration_bias_x", "acceleration_bias_y", "acceleration_bias_z"],
        "z_velocity_bias": ["z_velocity_bias"],
        "imu_acc": ["imu_acc_x", "imu_acc_y", "imu_acc_z"],
    }
    log_column_sources = {
        "z_deriv": ["local_pos_z_deriv", "z_deriv"],
    }

    dual_state_plot = (not delta_mode) and (state in dynamics_sources or state in euler_sources)
    acc_compare_plot = (not delta_mode) and state in ("acc_compare", "acc_diag")
    group_plot = (not delta_mode) and state in setpoint_groups
    if acc_compare_plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
        plt.subplots_adjust(bottom=0.12, hspace=0.35)
        plot_title = state
        axis_names = ["x", "y", "z"]
        velocity_indices = [STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]
        for axis_idx, group_ax in enumerate(axes):
            model_mask = finite_plot_mask(t, calc["acc_used"][:, axis_idx]) & active_plot_mask
            group_ax.plot(
                plot_t[model_mask],
                calc["acc_used"][model_mask, axis_idx],
                color="tab:cyan",
                linewidth=2.0,
                label=f"acc_used_{axis_names[axis_idx]}_dynamics",
            )
            thrust_mask = finite_plot_mask(t, calc["thrust_acc"][:, axis_idx]) & active_plot_mask
            group_ax.plot(
                plot_t[thrust_mask],
                calc["thrust_acc"][thrust_mask, axis_idx],
                color="tab:blue",
                linewidth=1.4,
                alpha=0.55,
                label=f"thrust_acc_{axis_names[axis_idx]}",
            )
            imu_mask = finite_plot_mask(t, calc["imu_acc"][:, axis_idx]) & active_plot_mask
            group_ax.plot(
                plot_t[imu_mask],
                calc["imu_acc"][imu_mask, axis_idx],
                color="tab:orange",
                linewidth=1.8,
                linestyle="--",
                label=f"livox_imu_acc_{axis_names[axis_idx]}_ned",
            )
            log_acc = np.full(len(actual), np.nan)
            dt_for_diff = np.diff(t)
            valid_dt = np.isfinite(dt_for_diff) & (dt_for_diff > 1.0e-9)
            vel_value = actual[:, velocity_indices[axis_idx]]
            valid_dt_indices = np.flatnonzero(valid_dt)
            log_acc[valid_dt_indices] = np.diff(vel_value)[valid_dt] / dt_for_diff[valid_dt]
            log_acc_mask = finite_plot_mask(plot_t, log_acc) & active_plot_mask
            group_ax.plot(
                plot_t[log_acc_mask],
                log_acc[log_acc_mask],
                color="tab:green",
                linewidth=1.3,
                alpha=0.75,
                label=f"dvel_{axis_names[axis_idx]}_dt",
            )
            if prediction_horizon > 1 and pred_calc_rollouts is not None:
                pred_segments = setpoint_prediction_segments(
                    plot_t,
                    prediction_start_indices,
                    pred_calc_rollouts,
                    "acc_used",
                    axis_idx,
                    args.dt,
                )
                add_segments(
                    group_ax,
                    pred_segments,
                    "tab:red",
                    f"acc_used_{axis_names[axis_idx]}_prediction_{prediction_horizon}step",
                    linewidth=1.5,
                    alpha=0.8,
                )
            group_ax.set_title(f"acc_{axis_names[axis_idx]}")
            group_ax.set_ylabel("m/s^2")
    elif group_plot:
        group_states = setpoint_groups[state]
        fig, axes = plt.subplots(len(group_states), 1, figsize=(12, 2.8 * len(group_states)), sharex=True)
        if len(group_states) == 1:
            axes = [axes]
        plt.subplots_adjust(bottom=0.12, hspace=0.35)
        plot_title = state
        for group_ax, group_state in zip(axes, group_states):
            key, axis, log_col = setpoint_sources[group_state]
            legend_base = setpoint_component_name(key, axis)
            model_mask = finite_plot_mask(t, calc[key][:, axis]) & active_plot_mask
            group_ax.plot(
                plot_t[model_mask], calc[key][model_mask, axis],
                color="tab:cyan", linewidth=2.0, label=f"{legend_base}_dynamics",
            )
            if prediction_horizon > 1 and pred_calc_rollouts is not None:
                pred_segments = setpoint_prediction_segments(
                    plot_t,
                    prediction_start_indices,
                    pred_calc_rollouts,
                    key,
                    axis,
                    args.dt,
                )
                add_segments(
                    group_ax,
                    pred_segments,
                    "tab:blue",
                    f"{legend_base}_prediction_{prediction_horizon}step",
                    linewidth=1.5,
                    alpha=0.85,
                )
            log_col = first_existing_column(df, log_col)
            if log_col is not None:
                log_value = df[log_col].to_numpy(float)
                log_plot_t = plot_time_for_log_column(df, log_col, plot_t, base_timestamp)
                log_mask = (
                    finite_plot_mask(log_plot_t, log_value)
                    & base_active_plot_mask
                    & plot_window_mask(log_plot_t, plot_start, plot_end)
                )
                group_ax.plot(
                    log_plot_t[log_mask], log_value[log_mask],
                    color="tab:orange", linewidth=2.0, linestyle="--", zorder=3,
                    label=f"{legend_base}_px4",
                )
            group_ax.set_title(legend_base)
            group_ax.set_ylabel("value")
    elif dual_state_plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
        ax = axes[0]
        plt.subplots_adjust(bottom=0.18)
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        axes = [ax]
        plt.subplots_adjust(bottom=0.18)
    plot_title = state

    if acc_compare_plot or group_plot:
        pass
    elif state in log_column_sources:
        if delta_mode:
            print("[ERROR] d-prefix is not supported for raw log columns")
            sys.exit(1)
        log_col = first_existing_column(df, log_column_sources[state])
        if log_col is None:
            print(f"[ERROR] no log column found for {state}")
            sys.exit(1)
        log_value = df[log_col].to_numpy(float)
        log_mask = finite_plot_mask(plot_t, log_value) & active_plot_mask
        plot_title = state
        ax.plot(
            plot_t[log_mask],
            log_value[log_mask],
            color="tab:orange",
            linewidth=2.0,
            linestyle="--",
            label=f"{log_col}_px4",
        )
        if state == "z_deriv":
            vz_idx = STATE_INDEX["vz"]
            z_idx = STATE_INDEX["z"]
            actual_vz = actual[:, vz_idx]
            actual_vz_mask = finite_plot_mask(plot_t, actual_vz) & active_plot_mask
            ax.plot(
                plot_t[actual_vz_mask],
                actual_vz[actual_vz_mask],
                color="tab:purple",
                linewidth=1.8,
                alpha=0.85,
                label="vel_z_actual",
            )
            dzdt = np.full(len(actual), np.nan)
            dt_for_diff = np.diff(t)
            valid_dt = np.isfinite(dt_for_diff) & (dt_for_diff > 1.0e-9)
            z_value = actual[:, z_idx]
            valid_dt_indices = np.flatnonzero(valid_dt)
            dzdt[valid_dt_indices] = np.diff(z_value)[valid_dt] / dt_for_diff[valid_dt]
            dzdt_mask = finite_plot_mask(plot_t, dzdt) & active_plot_mask
            ax.plot(
                plot_t[dzdt_mask],
                dzdt[dzdt_mask],
                color="tab:green",
                linewidth=1.6,
                alpha=0.8,
                label="dpos_z_dt",
            )
            if prediction_horizon > 1 and pred_rollouts is not None:
                value_func = lambda values, idx=vz_idx: values[:, idx]
                pred_segments, _ = state_prediction_segments(
                    plot_t,
                    prediction_start_indices,
                    pred_rollouts,
                    actual,
                    value_func,
                    args.dt,
                )
                add_segments(
                    ax,
                    pred_segments,
                    "tab:cyan",
                    f"vz_model_{prediction_horizon}step",
                    linewidth=1.5,
                    alpha=0.85,
                )
            else:
                model_vz = pred_next[:, vz_idx]
                model_mask = finite_plot_mask(t, model_vz) & active_plot_mask
                ax.plot(
                    plot_t[model_mask] + args.dt,
                    model_vz[model_mask],
                    color="tab:cyan",
                    linewidth=2.0,
                    label="vz_model",
                )
        ax.set_ylabel("velocity [m/s]")
    elif state in setpoint_sources:
        if delta_mode:
            print("[ERROR] d-prefix is only supported for states/euler angles, not setpoints")
            sys.exit(1)
        key, axis, log_col = setpoint_sources[state]
        legend_base = setpoint_component_name(key, axis)
        plot_title = legend_base
        model_mask = finite_plot_mask(t, calc[key][:, axis]) & active_plot_mask
        ax.plot(plot_t[model_mask], calc[key][model_mask, axis], color="tab:cyan", linewidth=2.0, label=f"{legend_base}_dynamics")
        if prediction_horizon > 1 and pred_calc_rollouts is not None:
            pred_segments = setpoint_prediction_segments(
                plot_t,
                prediction_start_indices,
                pred_calc_rollouts,
                key,
                axis,
                args.dt,
            )
            add_segments(
                ax,
                pred_segments,
                "tab:blue",
                f"{legend_base}_prediction_{prediction_horizon}step",
                linewidth=1.5,
                alpha=0.85,
            )
        log_col = first_existing_column(df, log_col)
        if log_col is not None:
            log_value = df[log_col].to_numpy(float)
            log_plot_t = plot_time_for_log_column(df, log_col, plot_t, base_timestamp)
            log_mask = (
                finite_plot_mask(log_plot_t, log_value)
                & base_active_plot_mask
                & plot_window_mask(log_plot_t, plot_start, plot_end)
            )
            ax.plot(
                log_plot_t[log_mask], log_value[log_mask],
                color="tab:orange", linewidth=2.0, linestyle="--", zorder=3,
                label=f"{legend_base}_px4",
            )
        ax.set_ylabel("setpoint")
    elif state in dynamics_sources:
        idx = dynamics_sources[state]
        if delta_mode:
            plot_title = f"d{state_name}"
            if prediction_horizon > 1 and pred_rollouts is not None:
                value_func = lambda values, idx=idx: values[:, idx]
                pred_segments, actual_segments = state_delta_segments(
                    plot_t,
                    prediction_start_indices,
                    pred_rollouts,
                    actual,
                    value_func,
                    args.dt,
                )
                add_segments(ax, pred_segments, "tab:cyan", f"d{state_name}_model_{prediction_horizon}step")
                add_segments(
                    ax,
                    actual_segments,
                    "tab:orange",
                    f"d{state_name}_actual_{prediction_horizon}step",
                    linestyle="--",
                )
            else:
                model_delta = pred_next[:, idx] - actual[:, idx]
                log_delta = actual_next[:, idx] - actual[:, idx]
                mask = finite_plot_mask(t, model_delta, log_delta) & active_plot_mask
                ax.plot(plot_t[mask], model_delta[mask], color="tab:cyan", linewidth=2.0, label=f"d{state_name}_model")
                ax.plot(plot_t[mask], log_delta[mask], color="tab:orange", linewidth=2.0, linestyle="--", label=f"d{state_name}_actual")
            ax.set_ylabel("")
        else:
            actual_value = actual[:, idx]
            actual_mask = finite_plot_mask(t, actual_value) & active_plot_mask
            plot_title = state_name
            ax.plot(plot_t[actual_mask], actual_value[actual_mask], color="tab:orange", linewidth=2.0, linestyle="--", label=f"{state_name}_actual")
            if prediction_horizon > 1 and pred_rollouts is not None:
                value_func = lambda values, idx=idx: values[:, idx]
                pred_segments, error_segments = state_prediction_segments(
                    plot_t,
                    prediction_start_indices,
                    pred_rollouts,
                    actual,
                    value_func,
                    args.dt,
                )
                add_segments(ax, pred_segments, "tab:cyan", f"{state_name}_model_{prediction_horizon}step")
            else:
                model_value = pred_next[:, idx]
                model_mask = finite_plot_mask(t, model_value) & active_plot_mask
                ax.plot(plot_t[model_mask] + args.dt, model_value[model_mask], color="tab:cyan", linewidth=2.0, label=f"{state_name}_model")
            ax.set_title(state_name)
            ax.set_ylabel("state")
            ax = axes[1]
            if prediction_horizon > 1 and pred_rollouts is not None:
                add_segments(ax, error_segments, "tab:red", f"{state_name}_model_actual_error")
            else:
                state_error = pred_next[:, idx] - actual_next[:, idx]
                error_mask = finite_plot_mask(t, state_error) & active_plot_mask
                ax.plot(
                    plot_t[error_mask] + args.dt,
                    state_error[error_mask],
                    color="tab:red",
                    linewidth=2.0,
                    label=f"{state_name}_model_actual_error",
                )
            ax.axhline(0.0, color="0.35", linewidth=1.0)
            ax.set_title(f"{state_name} error")
            ax.set_ylabel("model - actual")
    elif state in euler_sources:
        axis = euler_sources[state]
        pred_euler = np.array([quat_to_euler(q) for q in pred_next[:, 0:4]])
        actual_euler = np.array([quat_to_euler(q) for q in actual[:, 0:4]])
        actual_next_euler = np.array([quat_to_euler(q) if np.all(np.isfinite(q)) else [np.nan, np.nan, np.nan] for q in actual_next[:, 0:4]])
        if delta_mode:
            plot_title = f"d{state_name}"
            if prediction_horizon > 1 and pred_rollouts is not None:
                value_func = lambda values, axis=axis: np.array([quat_to_euler(q)[axis] for q in values[:, 0:4]])
                pred_segments, actual_segments = state_delta_segments(
                    plot_t,
                    prediction_start_indices,
                    pred_rollouts,
                    actual,
                    value_func,
                    args.dt,
                )
                add_segments(ax, pred_segments, "tab:cyan", f"d{state_name}_model_{prediction_horizon}step")
                add_segments(
                    ax,
                    actual_segments,
                    "tab:orange",
                    f"d{state_name}_actual_{prediction_horizon}step",
                    linestyle="--",
                )
            else:
                model_delta = pred_euler[:, axis] - actual_euler[:, axis]
                log_delta = actual_next_euler[:, axis] - actual_euler[:, axis]
                mask = finite_plot_mask(t, model_delta, log_delta) & active_plot_mask
                ax.plot(plot_t[mask], model_delta[mask], color="tab:cyan", linewidth=2.0, label=f"d{state_name}_model")
                ax.plot(plot_t[mask], log_delta[mask], color="tab:orange", linewidth=2.0, linestyle="--", label=f"d{state_name}_actual")
            ax.set_ylabel("")
        else:
            actual_value = actual_euler[:, axis]
            actual_mask = finite_plot_mask(t, actual_value) & active_plot_mask
            plot_title = state_name
            ax.plot(plot_t[actual_mask], actual_value[actual_mask], color="tab:orange", linewidth=2.0, linestyle="--", label=f"{state_name}_actual")
            if prediction_horizon > 1 and pred_rollouts is not None:
                value_func = lambda values, axis=axis: np.array([quat_to_euler(q)[axis] for q in values[:, 0:4]])
                pred_segments, error_segments = state_prediction_segments(
                    plot_t,
                    prediction_start_indices,
                    pred_rollouts,
                    actual,
                    value_func,
                    args.dt,
                )
                add_segments(ax, pred_segments, "tab:cyan", f"{state_name}_model_{prediction_horizon}step")
            else:
                model_value = pred_euler[:, axis]
                model_mask = finite_plot_mask(t, model_value) & active_plot_mask
                ax.plot(plot_t[model_mask] + args.dt, model_value[model_mask], color="tab:cyan", linewidth=2.0, label=f"{state_name}_model")
            ax.set_title(state_name)
            ax.set_ylabel("state")
            ax = axes[1]
            if prediction_horizon > 1 and pred_rollouts is not None:
                add_segments(ax, error_segments, "tab:red", f"{state_name}_model_actual_error")
            else:
                state_error = pred_euler[:, axis] - actual_next_euler[:, axis]
                error_mask = finite_plot_mask(t, state_error) & active_plot_mask
                ax.plot(
                    plot_t[error_mask] + args.dt,
                    state_error[error_mask],
                    color="tab:red",
                    linewidth=2.0,
                    label=f"{state_name}_model_actual_error",
                )
            ax.axhline(0.0, color="0.35", linewidth=1.0)
            ax.set_title(f"{state_name} error")
            ax.set_ylabel("model - actual")
    else:
        valid = sorted(
            set(dynamics_sources)
            | set(euler_sources)
            | set(setpoint_sources)
            | set(setpoint_groups)
            | set(log_column_sources)
            | {"acc_compare", "acc_diag"}
        )
        print("[ERROR] state must be one of:")
        for name in valid:
            print(" ", name)
        sys.exit(1)

    print(f"[INFO] csv={args.csv_path}")
    print(
        f"[INFO] state={state}, model_dt={args.dt:.6f}, csv_dt={csv_dt:.6f}, "
        f"rate_delay_steps={args.rate_delay_steps}, "
        f"motor_command_delay_steps={args.motor_command_delay_steps}, "
        f"horizon={prediction_horizon}, prediction_interval={args.prediction_interval:.3f}, "
        f"plot_window={plot_start:.3f}..{plot_end:.3f}s, "
        f"input_position_source={args.input_position_source}, "
        f"target_delay_steps={TARGET_LOCAL_SP_DELAY_STEPS}, "
        f"motor_input_source={args.motor_input_source}, "
        f"setpoint_prediction_state={args.setpoint_prediction_state}"
    )
    if prediction_horizon > 1:
        print(f"[INFO] recursive prediction starts: {len(prediction_start_indices)}")
    print(
        f"[INFO] plot samples: {int(np.count_nonzero(active_plot_mask))}/"
        f"{int(np.count_nonzero(base_active_plot_mask))} non-Idle in window"
    )
    if not dual_state_plot and not group_plot and not acc_compare_plot:
        ax.set_title(plot_title)
    for plot_ax in axes:
        plot_ax.set_xlabel("time [s]")
        plot_ax.grid()
        plot_ax.legend(fontsize=12)
        plot_ax.relim()
        plot_ax.autoscale_view()
        plot_ax.set_xlim(plot_start, plot_end)
        attach_interactive_navigation(fig, plot_ax)
    progress("plot ready; waiting for matplotlib window to close")
    plt.show()


if __name__ == "__main__":
    main()
