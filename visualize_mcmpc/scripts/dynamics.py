# 入力項目およびパラメータ一覧
#
# 実行時入力:
#   --csv                 読み込むログCSV。default: DEFAULT_CSV
#   --state               表示する状態。例: x, y, z, vx, vy, vz, wx, wy, wz, roll, pitch, yaw
#   --mode                予測モード。model_input / actuator_log / actual_state
#   --start-idx           予測開始行。--start-time指定時はそちらを優先
#   --start-time          表示時刻[s]で指定する単発予測開始時刻
#   --horizon             予測ステップ数。default: PREDICTION_HORIZON
#   --prediction-interval 複数予測線を描く間隔[s]。default: PREDICTION_INTERVAL
#   --plot-start          表示開始時刻[s]
#   --plot-end            表示終了時刻[s]
#   --auto-window         予測開始周辺だけに表示範囲を絞る
#   --full-range          auto-windowを無効化
#   --all-segments        時刻リセット後の最後の区間だけでなく全区間を表示
#
# model_inputで使う入力列:
#   target_x, target_y, target_z, target_yaw
#   target_x_delayed, target_y_delayed, target_z_delayed, target_yaw_delayed
#     - あれば delayed を優先
#   local_sp_x, local_sp_y, local_sp_z
#     - make_position_setpoint_from_row(..., use_local_sp=True)で使用
#   local_sp_vx, local_sp_vy, local_sp_vz
#   local_sp_ax, local_sp_ay, local_sp_az
#     - vel_int推定に使用。なければ内部推定を継続
#   rollspeed_integ, pitchspeed_integ, yawspeed_integ
#     - rate_int同期に使用。なければ内部推定を継続
#   hover_thrust, hover_thrust_valid
#     - あればhover thrustをログ値に同期
#   takeoff_state, takeoff_tilt_limit
#     - takeoff中のthrust_min/tilt_limit/no_thrust切り替えに使用
#   landed, maybe_landed, ground_contact
#     - no_thrust判定に使用
#
# actuator_logで使う入力列:
#   actuator_motor_0, actuator_motor_1, actuator_motor_2, actuator_motor_3
#   actuator_motor_input_0..3
#     - add_actuator_input_columns()が生成。あればこちらを優先
#
# actual_stateで使う入力列:
#   vel_x, vel_y, vel_z
#   angular_vel_x, angular_vel_y, angular_vel_z
#
import argparse
import copy
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib/")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning, )

DEFAULT_CSV = "/home/kt182/ws_mcmpc/src/quadcopter_mcmpc_expt/visualize_mcmpc/csv/offboard_control_log_real_sin_x.csv"

SIMULATION = False
# prediction mode values:
# - "actuator_log": use future logged actuator_motor values as model input.
# - "actual_state": use logged velocity/angular velocity to integrate position/attitude.
# - "model_input": use future model/controller inputs such as setpoints.
MODE_ACTUATOR_LOG = "actuator_log"
MODE_ACTUAL_STATE = "actual_state"
MODE_MODEL_INPUT = "model_input"

FUTURE_TARGET_COLUMNS = [
    "target_x",
    "target_y",
    "target_z",
    "target_yaw",
    "target_x_delayed",
    "target_y_delayed",
    "target_z_delayed",
    "target_yaw_delayed",
]

#Common Parameter
DT                                          = 0.02
A_OF_GRAVITY                                = 9.80665
CA_MINIMUM_YAW_MARGIN                       = 0.15 #PX4 allocatorのyaw用余裕
ANGULAR_ACCEL_LP                            = 30.0 #角速度微分のLPFカットオフ周波数
MPC_ACC_DECOUPLE                            = False #PX4の加速度-スラスト変換でz加速度を切り離すか
MOTOR_COMMAND_DELAY_STEPS                   = 0 #モータ指令がモデル反映までの遅れ
ACTUATOR_LOG_SHIFT_STEPS                    = -1 #actuator_logの時刻補正
TARGET_LOCAL_SP_DELAY_STEPS                 = 2 #target local position setpointの遅れ
RATE_INT_SYNC_PERIOD                        = 1.5 #rate controllerの積分項ログと内部推定を同期する周期
PREDICTION_HORIZON                          = 75 #予測ステップ数
PREDICTION_INTERVAL                         = 1.5 #予測時間
USE_RATE_INT_BIAS_TORQUE                    = True #Rate controllerの積分項に対するバイアス補正の有効化
USE_RATE_INT_BIAS_TORQUE_WITH_ACTUATOR_LOG  = True #Actuator logを用いたRate controllerの積分項に対するバイアス補正の有効化
USE_ACCELERATION_BIAS_OBSERVER              = True #モデル加速度とIMUから並進加速度バイアスを推定するか
USE_LIVOX_IMU_ACCELERATION                  = True #Livox IMUの加速度をバイアス観測に使用するか
USE_VELOCITY_DELTA_BIAS_OBSERVER            = True #1stepの速度差から加速度バイアスの推定の有効化
USE_ACCELERATION_BIAS_TREND_PREDICTION      = True #horizon内の加速度バイアスの変化傾向の予測の有効化
USE_Z_POSITION_VELOCITY_BIAS_CORRECTION     = True #Z軸位置と速度のバイアス補正の有効化
USE_DYNAMICS_BIAS_OBSERVER                  = False #motor_setpoint後のforce/torqueモデル誤差をログから推定するか
ACCELERATION_BIAS_ALPHA                     = 0.05 #加速度バイアスのLPF係数
VELOCITY_DELTA_BIAS_ALPHA                   = 0.10 #速度差バイアスのLPF係数
Z_POSITION_VELOCITY_BIAS_ALPHA              = 0.05 #Z軸位置と速度のバイアス補正のLPF係数
Z_POSITION_VELOCITY_BIAS_LIMIT              = 0.5 #Z軸位置と速度のバイアス補正の制限
DYNAMICS_FORCE_BIAS_ALPHA                   = 0.08 #force_bias observerのLPF係数
DYNAMICS_TORQUE_BIAS_ALPHA                  = 0.08 #torque_bias observerのLPF係数
ACCELERATION_BIAS_TREND_LOOKBACK_STEPS      = 10 #加速度バイアスの変化傾向の予測に使用する過去ステップ数
ACCELERATION_BIAS_TREND_RATE_LIMIT          = 0.30 #加速度バイアスの変化率の制限
ACCELERATION_BIAS_LIMIT                     = np.array([3.0, 3.0, 2.0], dtype=float)
ACCELERATION_BIAS_DEADBAND                  = np.array([0.02, 0.02, 0.01], dtype=float)
DYNAMICS_FORCE_BIAS_LIMIT                   = np.array([8.0, 8.0, 8.0], dtype=float)
DYNAMICS_TORQUE_BIAS_LIMIT                  = np.array([0.8, 0.8, 0.25], dtype=float)
DYNAMICS_FORCE_BIAS_DEADBAND                = np.array([0.04, 0.04, 0.02], dtype=float)
DYNAMICS_TORQUE_BIAS_DEADBAND               = np.array([0.02, 0.02, 0.01], dtype=float)
VELOCITY_DELTA_BIAS_AXES                    = np.array([True, True, True], dtype=bool) #速度差からのバイアス推定を有効にする軸
TRANSLATION_BIAS_AXES                       = np.array([True, True, True], dtype=bool) #推定した並進バイアスを実際の加速度計算へ入れる軸
DYNAMICS_FORCE_BIAS_AXES                    = np.array([True, True, True], dtype=bool)
DYNAMICS_TORQUE_BIAS_AXES                   = np.array([True, True, True], dtype=bool)
LINEAR_VELOCITY_DAMPING                     = np.zeros(3, dtype=float) #NED速度に比例する減衰[1/s]
ANGULAR_VELOCITY_DAMPING                    = np.zeros(3, dtype=float) #body角速度に比例する減衰[Nm/(rad/s)]
BODY_TORQUE_SCALE                           = np.ones(3, dtype=float) #motor差分から出る実効トルク倍率
TAKEOFF_STATE_DISARMED                      = 1
TAKEOFF_STATE_SPOOLUP                       = 2
TAKEOFF_STATE_READY_FOR_TAKEOFF             = 3
TAKEOFF_STATE_RAMPUP                        = 4
TAKEOFF_STATE_FLIGHT                        = 5

if SIMULATION:
    MPC_XY_P                    = 0.30
    MPC_Z_P                     = 1.00
    MPC_XY_VEL_P_ACC            = 1.80
    MPC_XY_VEL_I_ACC            = 0.40
    MPC_XY_VEL_D_ACC            = 0.20
    MPC_Z_VEL_P_ACC             = 4.00
    MPC_Z_VEL_I_ACC             = 2.00
    MPC_Z_VEL_D_ACC             = 0.00
    MPC_XY_VEL_MAX              = 12.0
    MPC_Z_VEL_MAX_UP            = 3.0
    MPC_Z_VEL_MAX_DOWN          = 1.0
    MPC_THR_HOVER               = 0.6295
    MPC_THR_MIN                 = 0.10
    MPC_THR_MAX                 = 0.90
    MPC_THR_XY_MARGIN           = 0.30
    MPC_TILT_MAX                = np.deg2rad(45.0)
    MPC_VELD_LP                 = 5.0

    MC_ROLL_P                   = 3.30
    MC_PITCH_P                  = 3.30
    MC_YAW_P                    = 2.80
    MC_YAW_WEIGHT               = 0.50
    MC_ROLLRATE_MAX             = np.deg2rad(220.0)
    MC_PITCHRATE_MAX            = np.deg2rad(220.0)
    MC_YAWRATE_MAX              = np.deg2rad(200.0)
    MC_ROLLRATE_P               = 0.150
    MC_PITCHRATE_P              = 0.150
    MC_YAWRATE_P                = 0.500
    MC_ROLLRATE_K               = 1.0
    MC_PITCHRATE_K              = 1.0
    MC_YAWRATE_K                = 1.0
    MC_ROLLRATE_D               = 0.0035
    MC_PITCHRATE_D              = 0.0035
    MC_YAWRATE_D                = 0.00
    MC_ROLLRATE_I               = 0.20
    MC_PITCHRATE_I              = 0.20
    MC_YAWRATE_I                = 0.100
    MC_ROLLRATE_FF              = 0.0
    MC_PITCHRATE_FF             = 0.0
    MC_YAWRATE_FF               = 0.0
    MC_RR_INT_LIM               = 0.30
    MC_PR_INT_LIM               = 0.30
    MC_YR_INT_LIM               = 0.30
    MC_YAW_TQ_CUTOFF            = 2.0
    DRONE_MASS                  = 2.3
    DRONE_INERTIA               = np.diag([0.0418, 0.04226, 0.05619])
    # MOTOR MODEL: quadratic motor setpoint to rad/s approximation
    MOTOR_SPEED_MODEL           = "quadratic"
    MOTOR_INPUT_SCALING         = 1000.0 #SDF input_scaling: actuator motorからrad/sへのスケーリング
    MAX_ROT_VELOCITY            = 1121.83 #SDF maxRotVelocity [rad/s]
    MOTOR_TIME_CONSTANT_UP      = 0.0125 #SDF timeConstantUp
    MOTOR_TIME_CONSTANT_DOWN    = 0.025 #SDF timeConstantDown
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
    MPC_THR_HOVER = 0.46 #実機ログで最も合うため固定
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

    # MOTOR MODEL: real vehicle quadratic approximation
    MOTOR_SPEED_MODEL           = "quadratic"
    MOTOR_INPUT_SCALING         = 1180.0
    MAX_ROT_VELOCITY            = 1121.83
    MOTOR_TIME_CONSTANT_UP      = 0.004 #要同定: モータ回転数が上がる際の時定数
    MOTOR_TIME_CONSTANT_DOWN    = 0.070 #要同定: モータ回転数が下がる際の時定数
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
# actuator min and max value for PX4
PX4_ACTUATOR_MIN            = np.zeros(4, dtype=float)
PX4_ACTUATOR_MAX            = np.ones(4, dtype=float)
MOTOR_THRUST_CONSTANT       = 1.0792e-5
MOMENT_CONSTANT             = 1.51e-7

ZERO3 = np.zeros(3, dtype=float)
LINEAR_VELOCITY_DAMPING_ENABLED = bool(np.any(LINEAR_VELOCITY_DAMPING != 0.0))
ANGULAR_VELOCITY_DAMPING_ENABLED = bool(np.any(ANGULAR_VELOCITY_DAMPING != 0.0))
BODY_TORQUE_SCALE_ENABLED = bool(np.any(BODY_TORQUE_SCALE != 1.0))
DRONE_INERTIA_DIAG = np.diag(DRONE_INERTIA).astype(float)
DRONE_INERTIA_IS_DIAGONAL = bool(np.allclose(DRONE_INERTIA, np.diag(DRONE_INERTIA_DIAG)))
DRONE_INERTIA_INV_DIAG = 1.0 / DRONE_INERTIA_DIAG

STATE_NAMES = [
    "e0", "e1", "e2", "e3",
    "wx", "wy", "wz",
    "x", "y", "z",
    "vx", "vy", "vz",
]
STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}

def progress(message):
    print(f"[PROGRESS] {message}", flush=True)

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

def integrate_quat(q, omega_body, dt):
    q = normalize(q)
    wx, wy, wz = omega_body
    omega_quat = np.array([0.0, wx, wy, wz], dtype=float)
    q_dot = 0.5 * quat_multiply(q, omega_quat)
    return normalize(q + q_dot * dt)

def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)

def preprocess_log(df, step_dt):
    df = df.copy() #csv log
    df.columns = df.columns.str.strip() #remove whitespace from column names
    # 1. control_time_sの有効な行のみ残す
    df = df[np.isfinite(df["control_time_s"].to_numpy(float))].reset_index(drop=True)

    # 2. 時刻が戻る箇所があればsegmentとして扱う
    t = df["control_time_s"].to_numpy(float)
    df["_raw_control_time_s"] = t
    reset_indices = np.flatnonzero(np.diff(t) <= 0.0) + 1
    starts = np.r_[0, reset_indices] #離陸からhover, 移動開始まで
    ends = np.r_[reset_indices, len(df)] #移動開始から最後まで

    segment_id = np.zeros(len(df), dtype=np.int64)
    segment_time = np.zeros(len(df), dtype=float)
    for segment_idx, (start, end) in enumerate(zip(starts, ends)):
        segment_id[start:end] = segment_idx
        segment_time[start:end] = t[start:end] - t[start]
    df["_segment_id"] = segment_id
    df["_segment_time_s"] = segment_time

    # 3. 時刻リセットがある場合は連続時間に貼り直す
    if len(starts) > 1:
        continuous_t = np.empty_like(t)
        next_start_time = 0.0
        for start, end in zip(starts, ends):
            segment_t = t[start:end]
            continuous_t[start:end] = next_start_time + segment_t - segment_t[0]
            next_start_time = continuous_t[end - 1] + step_dt
        df["control_time_s"] = continuous_t

    # 4. dt刻みに丸めて重複サンプルを落とす
    t = df["control_time_s"].to_numpy(float)
    t0 = float(t[0])
    sample_index = np.rint((t - t0) / step_dt).astype(np.int64)

    df["_sample_index"] = sample_index
    df = (
        df.drop_duplicates("_sample_index", keep="last")
          .sort_values("_sample_index")
    )
    full_sample_index = np.arange(
        int(df["_sample_index"].min()),
        int(df["_sample_index"].max()) + 1,
        dtype=np.int64,
    )
    df = df.set_index("_sample_index").reindex(full_sample_index)
    df.index.name = "_sample_index"

    discrete_cols = [
        "_segment_id",
        "phase",
        "event",
        "takeoff_state",
        "landed",
        "ground_contact",
        "maybe_landed",
        "hover_thrust_valid",
    ]
    for col in discrete_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    interpolate_cols = [col for col in numeric_cols if col not in discrete_cols]
    if interpolate_cols:
        df[interpolate_cols] = (
            df[interpolate_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    object_cols = [col for col in df.columns if col not in numeric_cols]
    if object_cols:
        df[object_cols] = df[object_cols].ffill().bfill()
    if "_segment_id" in df.columns:
        df["_segment_id"] = np.rint(df["_segment_id"].to_numpy(float)).astype(np.int64)
    df = df.reset_index()

    # 5. 時刻を 0, dt, 2dt... にそろえる
    df["control_time_s"] = df["_sample_index"].to_numpy(float) * step_dt
    df = df.drop(columns=["_sample_index"])

    return df

#予測に使う actuator_motor 入力専用列 (actuator_log)
def add_actuator_input_columns(df, shift_steps=ACTUATOR_LOG_SHIFT_STEPS):
    df = df.copy()
    actuator_cols =  [f"actuator_motor_{motor_idx}" for motor_idx in range(4)]
    input_cols = [f"actuator_motor_input_{i}" for i in range(4)]
    if not all(col in df.columns for col in actuator_cols):
        return df

    # segmentがあるならsegmentごとにshiftする
    if "_segment_id" in df.columns:
        shifted = df.groupby("_segment_id", sort=False)[actuator_cols].shift(shift_steps)
    else:
        shifted = df[actuator_cols].shift(shift_steps)

    # 予測入力用の列として追加
    for src, dst in zip(actuator_cols, input_cols):
        df[dst] = shifted[src]
    return df

# モデル用状態ベクトルの作成
def state_from_row(row):
    state = np.zeros(len(STATE_NAMES), dtype=float)
    state[0:4] = euler_to_quat(row["roll"], row["pitch"], row["yaw"])
    #角速度
    state[STATE_INDEX["wx"]] = row["angular_vel_x"]
    state[STATE_INDEX["wy"]] = row["angular_vel_y"]
    state[STATE_INDEX["wz"]] = row["angular_vel_z"]
    # 位置
    state[STATE_INDEX["x"]] = row["pos_x"]
    state[STATE_INDEX["y"]] = row["pos_y"]
    state[STATE_INDEX["z"]] = row["pos_z"]
    # 速度
    state[STATE_INDEX["vx"]] = row["vel_x"]
    state[STATE_INDEX["vy"]] = row["vel_y"]
    state[STATE_INDEX["vz"]] = row["vel_z"]
    return state


RATE_INT_COLUMNS = ["rollspeed_integ", "pitchspeed_integ", "yawspeed_integ"]


def logged_rate_int_from_row(row):
    if all(col in row.index for col in RATE_INT_COLUMNS):
        logged_rate_int = row[RATE_INT_COLUMNS].to_numpy(float)
        if np.all(np.isfinite(logged_rate_int)):
            return logged_rate_int
    return None


def estimate_vel_int_from_logged_setpoint(row, state, prev_acc):
    required = [
        "local_sp_vx",
        "local_sp_vy",
        "local_sp_vz",
        "local_sp_ax",
        "local_sp_ay",
        "local_sp_az",
    ]
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


def make_position_setpoint_from_row(row, horizon_step=None, use_local_sp=False):
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

    pos_sp = np.array([x_ref, y_ref, z_ref], dtype=float)
    return pos_sp, yaw_sp


#csvの1行から4つのモータ指令の取り出し
def actuator_motor_from_row(row):
    input_cols = [f"actuator_motor_input_{i}" for i in range(4)]
    raw_cols = [f"actuator_motor_{i}" for i in range(4)]
    for cols in (input_cols, raw_cols):
        if not all(col in row.index for col in cols):
            continue
        actuator = row[cols].to_numpy(float)
        if np.all(np.isfinite(actuator)):
            return np.clip(actuator, 0.0, 1.0)
    return None

def motor_speed_ref_from_setpoint(motor_setpoint):
    u = np.clip(np.asarray(motor_setpoint, dtype=float)[:4], 0.0, 1.0)
    motor_speed_ref = (
        1528.43944677 * u
        - 406.61258522 * u * u
    )
    return np.clip(motor_speed_ref, 0.0, MAX_ROT_VELOCITY)

def motor_speed_from_setpoint(motor_setpoint, step_dt, prev_motor_speed=None):
    if MOTOR_SPEED_MODEL == "quadratic":
        motor_speed_ref = motor_speed_ref_from_setpoint(motor_setpoint)
    else:
        raise ValueError(f"unknown MOTOR_SPEED_MODEL: {MOTOR_SPEED_MODEL}")
    if prev_motor_speed is None:
        return motor_speed_ref
    prev_motor_speed = np.clip(
        np.asarray(prev_motor_speed, dtype=float)[:4],
        0.0,
        MAX_ROT_VELOCITY,
    )
    tau = np.where(
        motor_speed_ref > prev_motor_speed,
        MOTOR_TIME_CONSTANT_UP,
        MOTOR_TIME_CONSTANT_DOWN,
    )
    # motor 1次遅れ
    alpha = np.exp(-step_dt / np.maximum(tau, 1.0e-9))
    return alpha * prev_motor_speed + (1.0 - alpha) * motor_speed_ref

#4つのロータ角速度[rad/s]から機体にかかる力, トルクの計算
def motor_speed_to_force_torque(motor_speed):
    motor_speed = np.asarray(motor_speed, dtype=float)[:4]
    # 各ロータ推力: F = k_f * omega^2
    motor_speed_squared = motor_speed * motor_speed
    rotor_forces = MOTOR_THRUST_CONSTANT * motor_speed_squared
    # FRD body座標では、上向き推力は body z負方向
    force_body = np.array([
        0.0,
        0.0,
        -np.sum(rotor_forces),
    ], dtype=float)
    torque_body = np.array([
        -np.dot(ROTOR_POSITIONS[:, 1], rotor_forces),
        np.dot(ROTOR_POSITIONS[:, 0], rotor_forces),
        MOMENT_CONSTANT * np.dot(ROTOR_YAW_SIGNS, motor_speed_squared),
    ], dtype=float)
    if BODY_TORQUE_SCALE_ENABLED:
        torque_body *= BODY_TORQUE_SCALE
    return rotor_forces, force_body, torque_body


def torque_from_motor_setpoint_no_lag(motor_setpoint):
    motor_speed = motor_speed_ref_from_setpoint(motor_setpoint)
    motor_speed_squared = motor_speed * motor_speed
    rotor_forces = MOTOR_THRUST_CONSTANT * motor_speed_squared
    torque_body = np.array([
        -np.dot(ROTOR_POSITIONS[:, 1], rotor_forces),
        np.dot(ROTOR_POSITIONS[:, 0], rotor_forces),
        MOMENT_CONSTANT * np.dot(ROTOR_YAW_SIGNS, motor_speed_squared),
    ], dtype=float)
    if BODY_TORQUE_SCALE_ENABLED:
        torque_body *= BODY_TORQUE_SCALE
    return torque_body


def angular_acceleration_from_torque(torque_body, omega):
    if DRONE_INERTIA_IS_DIAGONAL:
        inertia_omega = DRONE_INERTIA_DIAG * omega
        cross_term = np.array([
            omega[1] * inertia_omega[2] - omega[2] * inertia_omega[1],
            omega[2] * inertia_omega[0] - omega[0] * inertia_omega[2],
            omega[0] * inertia_omega[1] - omega[1] * inertia_omega[0],
        ], dtype=float)
        rhs = torque_body - cross_term
        if ANGULAR_VELOCITY_DAMPING_ENABLED:
            rhs = rhs - ANGULAR_VELOCITY_DAMPING * omega
        return rhs * DRONE_INERTIA_INV_DIAG

    inertia_omega = DRONE_INERTIA @ omega
    angular_damping_torque = (
        ANGULAR_VELOCITY_DAMPING * omega
        if ANGULAR_VELOCITY_DAMPING_ENABLED
        else ZERO3
    )
    return np.linalg.solve(
        DRONE_INERTIA,
        torque_body - angular_damping_torque - np.cross(omega, inertia_omega),
    )

# roll, pitch, yaw, thrust_z の制御入力から4つのモータ指令を計算する
def allocate_px4_quad_x(control_sp):
    control_sp = np.asarray(control_sp, dtype=float)
    motor_setpoint = allocate_px4_quad_x_motor_setpoint(control_sp)
    allocated_control = PX4_QUAD_X_MIX_INV @ motor_setpoint
    unallocated_control = control_sp - allocated_control
    return motor_setpoint, unallocated_control


def allocate_px4_quad_x_motor_setpoint(control_sp):
    control_sp = np.asarray(control_sp, dtype=float)
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
        motor_raw = desaturate_motor_outputs(
            motor_raw,
            desaturation_vector,
            increase_only,
            actuator_max,
        )
    motor_raw += yaw_mix * control_sp[2]
    yaw_actuator_max = PX4_ACTUATOR_MAX + (
        PX4_ACTUATOR_MAX - PX4_ACTUATOR_MIN
    ) * CA_MINIMUM_YAW_MARGIN
    desaturation_steps = [
        (yaw_mix, False, yaw_actuator_max),
        (thrust_z_mix, True, PX4_ACTUATOR_MAX),
    ]
    for desaturation_vector, increase_only, actuator_max in desaturation_steps:
        motor_raw = desaturate_motor_outputs(
            motor_raw,
            desaturation_vector,
            increase_only,
            actuator_max,
        )
    motor_setpoint = np.clip(motor_raw, PX4_ACTUATOR_MIN, PX4_ACTUATOR_MAX)
    return motor_setpoint

#モータ出力が 0〜1 の範囲を超えたときに、指定した方向へ全体を動かして飽和を減らす関数
def desaturate_motor_outputs(motor_raw, desaturation_vector, increase_only, actuator_max):
    motor_raw = np.asarray(motor_raw, dtype=float).copy()
    k_min = 0.0
    k_max = 0.0
    for value, desat, minimum, maximum in zip(
        motor_raw,
        desaturation_vector,
        PX4_ACTUATOR_MIN,
        actuator_max,
    ):
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
    for value, desat, minimum, maximum in zip(
        motor_raw,
        desaturation_vector,
        PX4_ACTUATOR_MIN,
        actuator_max,
    ):
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

def estimate_rate_int_bias_torque(rate_int, thrust_z_setpoint, step_dt):
    trim_control = np.array([
        rate_int[0],
        rate_int[1],
        rate_int[2],
        thrust_z_setpoint,
    ], dtype=float)
    trim_motor = allocate_px4_quad_x_motor_setpoint(trim_control)
    return torque_from_motor_setpoint_no_lag(trim_motor)

def should_apply_rate_int_bias_torque(using_actuator_motor_override):
    if not USE_RATE_INT_BIAS_TORQUE:
        return False
    return (not using_actuator_motor_override) or USE_RATE_INT_BIAS_TORQUE_WITH_ACTUATOR_LOG

def apply_motor_command_delay(motor_setpoint, motor_command_delay_buffer, motor_command_delay_steps):
    motor_setpoint = np.asarray(motor_setpoint, dtype=float).copy()
    if motor_command_delay_buffer is None or motor_command_delay_steps <= 0:
        return motor_setpoint

    while len(motor_command_delay_buffer) < motor_command_delay_steps:
        motor_command_delay_buffer.append(motor_setpoint.copy())
    motor_command_delay_buffer.append(motor_setpoint.copy())
    return motor_command_delay_buffer.pop(0)

def dynamics_step(state, input_data, dt, prev_motor_speed=None):
    state = np.asarray(state, dtype=float).copy()
    q = normalize(state[0:4])
    omega = state[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]].copy()
    pos = state[[STATE_INDEX["x"], STATE_INDEX["y"], STATE_INDEX["z"]]].copy()
    vel = state[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]].copy()
    kind = input_data["kind"]
    debug = {}
    if kind == "actual_state":
        vel_next = np.asarray(input_data["velocity"], dtype=float)
        omega_next = np.asarray(input_data["angular_velocity"], dtype=float)
        pos_next = pos + vel_next * dt
        q_next = integrate_quat(q, omega_next, dt)
        acc_ned = np.full(3, np.nan)
        omega_dot = np.full(3, np.nan)

    elif kind == "hold":
        vel_next = vel.copy()
        omega_next = omega.copy()
        pos_next = pos.copy()
        q_next = q.copy()
        acc_ned = np.zeros(3, dtype=float)
        omega_dot = np.zeros(3, dtype=float)
        debug.update({
            "thrust_acc": np.zeros(3, dtype=float),
            "acc_used": acc_ned,
            "force_body": np.zeros(3, dtype=float),
            "torque_body": np.zeros(3, dtype=float),
            "motor_speed": np.zeros(4, dtype=float),
        })

    elif kind == "actuator":
        actuator = np.asarray(input_data["actuator"], dtype=float)
        motor_speed = motor_speed_from_setpoint(
            actuator,
            dt,
            prev_motor_speed=prev_motor_speed,
        )
        rotor_forces, force_body, torque_body = motor_speed_to_force_torque(motor_speed)
        force_body_raw = force_body.copy()
        torque_body_raw = torque_body.copy()
        force_bias = np.asarray(
            input_data.get("force_bias", np.zeros(3, dtype=float)),
            dtype=float,
        )
        bias_torque = np.asarray(
            input_data.get("bias_torque", np.zeros(3, dtype=float)),
            dtype=float,
        )
        torque_bias = np.asarray(
            input_data.get("torque_bias", np.zeros(3, dtype=float)),
            dtype=float,
        )
        force_body = force_body_raw + force_bias
        torque_body = torque_body_raw - bias_torque + torque_bias
        thrust_acc_ned = (
            quat_to_rotmat(q) @ force_body / DRONE_MASS
            + np.array([0.0, 0.0, A_OF_GRAVITY], dtype=float)
        )
        translation_bias = np.asarray(
            input_data.get("acceleration_bias", np.zeros(3, dtype=float)),
            dtype=float,
        ).copy()
        translation_bias[~TRANSLATION_BIAS_AXES] = 0.0
        if LINEAR_VELOCITY_DAMPING_ENABLED:
            linear_damping_acc = -LINEAR_VELOCITY_DAMPING * vel
            acc_ned = thrust_acc_ned + translation_bias + linear_damping_acc
        else:
            linear_damping_acc = ZERO3
            acc_ned = thrust_acc_ned + translation_bias
        angular_damping_torque = (
            ANGULAR_VELOCITY_DAMPING * omega
            if ANGULAR_VELOCITY_DAMPING_ENABLED
            else ZERO3
        )
        omega_dot = angular_acceleration_from_torque(torque_body, omega)
        vel_next = vel + acc_ned * dt
        pos_next = pos + vel_next * dt
        omega_next = omega + omega_dot * dt
        q_next = integrate_quat(q, omega_next, dt)
        debug.update({
            "motor_speed": motor_speed,
            "rotor_forces": rotor_forces,
            "force_body": force_body,
            "force_body_raw": force_body_raw,
            "force_bias": force_bias,
            "torque_body": torque_body,
            "torque_body_raw": torque_body_raw,
            "bias_torque": bias_torque,
            "torque_bias": torque_bias,
            "thrust_acc": thrust_acc_ned,
            "acceleration_bias": translation_bias,
            "linear_damping_acc": linear_damping_acc,
            "angular_damping_torque": angular_damping_torque,
            "acc_used": acc_ned,
        })
    elif kind == "force_torque":
        force_body = np.asarray(input_data["force_body"], dtype=float)
        torque_body = np.asarray(input_data["torque_body"], dtype=float)
        force_body_raw = force_body.copy()
        torque_body_raw = torque_body.copy()
        force_bias = np.asarray(
            input_data.get("force_bias", np.zeros(3, dtype=float)),
            dtype=float,
        )
        torque_bias = np.asarray(
            input_data.get("torque_bias", np.zeros(3, dtype=float)),
            dtype=float,
        )
        force_body = force_body_raw + force_bias
        torque_body = torque_body_raw + torque_bias
        thrust_acc_ned = (
            quat_to_rotmat(q) @ force_body / DRONE_MASS
            + np.array([0.0, 0.0, A_OF_GRAVITY], dtype=float)
        )
        translation_bias = np.asarray(
            input_data.get("acceleration_bias", np.zeros(3, dtype=float)),
            dtype=float,
        ).copy()
        translation_bias[~TRANSLATION_BIAS_AXES] = 0.0
        if LINEAR_VELOCITY_DAMPING_ENABLED:
            linear_damping_acc = -LINEAR_VELOCITY_DAMPING * vel
            acc_ned = thrust_acc_ned + translation_bias + linear_damping_acc
        else:
            linear_damping_acc = ZERO3
            acc_ned = thrust_acc_ned + translation_bias
        angular_damping_torque = (
            ANGULAR_VELOCITY_DAMPING * omega
            if ANGULAR_VELOCITY_DAMPING_ENABLED
            else ZERO3
        )
        omega_dot = angular_acceleration_from_torque(torque_body, omega)
        vel_next = vel + acc_ned * dt
        pos_next = pos + vel_next * dt
        omega_next = omega + omega_dot * dt
        q_next = integrate_quat(q, omega_next, dt)
        debug.update({
            "force_body": force_body,
            "force_body_raw": force_body_raw,
            "force_bias": force_bias,
            "torque_body": torque_body,
            "torque_body_raw": torque_body_raw,
            "torque_bias": torque_bias,
            "thrust_acc": thrust_acc_ned,
            "acceleration_bias": translation_bias,
            "linear_damping_acc": linear_damping_acc,
            "angular_damping_torque": angular_damping_torque,
            "acc_used": acc_ned,
        })
    else:
        raise ValueError(f"unknown dynamics input kind: {kind}")
    next_state = state.copy()
    next_state[0:4] = q_next
    next_state[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]] = omega_next
    next_state[[STATE_INDEX["x"], STATE_INDEX["y"], STATE_INDEX["z"]]] = pos_next
    next_state[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]] = vel_next
    debug.update({
        "acc_ned": acc_ned,
        "omega_dot": omega_dot,
    })
    return next_state, debug


def log_vector_from_row(row, names):
    if not all(name in row.index for name in names):
        return None
    values = row[list(names)].to_numpy(float)
    if not np.all(np.isfinite(values)):
        return None
    return values


def update_dynamics_bias_observer(context, row, next_row, state, debug, dt):
    if not USE_DYNAMICS_BIAS_OBSERVER or next_row is None or dt <= 0.0:
        return {}
    if "force_bias" not in context:
        context["force_bias"] = np.zeros(3, dtype=float)
    if "torque_bias" not in context:
        context["torque_bias"] = np.zeros(3, dtype=float)

    observer_debug = {}
    current_vel = log_vector_from_row(row, ("vel_x", "vel_y", "vel_z"))
    next_vel = log_vector_from_row(next_row, ("vel_x", "vel_y", "vel_z"))
    model_acc = debug.get("acc_used")
    if current_vel is not None and next_vel is not None and model_acc is not None:
        model_acc = np.asarray(model_acc, dtype=float)
        if np.all(np.isfinite(model_acc)):
            logged_acc = (next_vel - current_vel) / dt
            acc_residual = logged_acc - model_acc
            acc_residual[np.abs(acc_residual) < DYNAMICS_FORCE_BIAS_DEADBAND] = 0.0
            force_correction = quat_to_rotmat(state[0:4]).T @ (DRONE_MASS * acc_residual)
            force_correction[~DYNAMICS_FORCE_BIAS_AXES] = 0.0
            context["force_bias"] += DYNAMICS_FORCE_BIAS_ALPHA * force_correction
            context["force_bias"] = np.clip(
                context["force_bias"],
                -DYNAMICS_FORCE_BIAS_LIMIT,
                DYNAMICS_FORCE_BIAS_LIMIT,
            )
            observer_debug.update({
                "logged_acc": logged_acc,
                "force_bias_residual_acc": acc_residual,
                "force_bias_correction": force_correction,
                "force_bias_observed": context["force_bias"].copy(),
            })

    current_omega = log_vector_from_row(
        row,
        ("angular_vel_x", "angular_vel_y", "angular_vel_z"),
    )
    next_omega = log_vector_from_row(
        next_row,
        ("angular_vel_x", "angular_vel_y", "angular_vel_z"),
    )
    model_omega_dot = debug.get("omega_dot")
    if (
        current_omega is not None
        and next_omega is not None
        and model_omega_dot is not None
    ):
        model_omega_dot = np.asarray(model_omega_dot, dtype=float)
        if np.all(np.isfinite(model_omega_dot)):
            logged_omega_dot = (next_omega - current_omega) / dt
            omega_dot_residual = logged_omega_dot - model_omega_dot
            omega_dot_residual[np.abs(omega_dot_residual) < DYNAMICS_TORQUE_BIAS_DEADBAND] = 0.0
            torque_correction = DRONE_INERTIA @ omega_dot_residual
            torque_correction[~DYNAMICS_TORQUE_BIAS_AXES] = 0.0
            context["torque_bias"] += DYNAMICS_TORQUE_BIAS_ALPHA * torque_correction
            context["torque_bias"] = np.clip(
                context["torque_bias"],
                -DYNAMICS_TORQUE_BIAS_LIMIT,
                DYNAMICS_TORQUE_BIAS_LIMIT,
            )
            observer_debug.update({
                "logged_omega_dot": logged_omega_dot,
                "torque_bias_residual_omega_dot": omega_dot_residual,
                "torque_bias_correction": torque_correction,
                "torque_bias_observed": context["torque_bias"].copy(),
            })
    return observer_debug


def next_row_for_observer(df, row_idx):
    next_idx = row_idx + 1
    if next_idx >= len(df):
        return None
    if (
        "_segment_id" in df.columns
        and df.loc[next_idx, "_segment_id"] != df.loc[row_idx, "_segment_id"]
    ):
        return None
    return df.loc[next_idx]

def make_input_from_row(row, state, mode, context, dt):
    if mode == MODE_ACTUATOR_LOG:
        return make_input_from_logged_actuator(row, state, context, dt)
    if mode == MODE_ACTUAL_STATE:
        return make_input_from_actual_state(row), {}, context
    if mode == MODE_MODEL_INPUT:
        return make_input_from_setpoint(row, state, context, dt)
    raise ValueError(f"unknown mode: {mode}")

def make_input_from_logged_actuator(row, state, context, dt, horizon_step=None, use_log_feedback=True):
    actuator = actuator_motor_from_row(row)
    if actuator is None:
        raise ValueError("actuator_motor input is missing or invalid")
    input_data, input_debug, context = make_input_from_setpoint(
        row,
        state,
        context,
        dt,
        horizon_step=horizon_step,
        use_log_feedback=use_log_feedback,
    )
    calculated_actuator = np.asarray(
        input_data.get("actuator", np.zeros(4, dtype=float)),
        dtype=float,
    ).copy()
    input_debug["calculated_motor_setpoint_used"] = np.asarray(
        calculated_actuator,
        dtype=float,
    )
    input_data = {
        "kind": "actuator",
        "actuator": actuator,
        "bias_torque": input_data.get("bias_torque", np.zeros(3, dtype=float)),
        "acceleration_bias": input_data.get("acceleration_bias", np.zeros(3, dtype=float)),
        "force_bias": input_data.get("force_bias", np.zeros(3, dtype=float)),
        "torque_bias": input_data.get("torque_bias", np.zeros(3, dtype=float)),
    }
    input_debug["motor_setpoint_used"] = actuator.copy()
    return input_data, input_debug, context

def make_input_from_actual_state(row):
    return {
        "kind": "actual_state",
        "velocity": np.array([
            row["vel_x"],
            row["vel_y"],
            row["vel_z"],
        ], dtype=float),
        "angular_velocity": np.array([
            row["angular_vel_x"],
            row["angular_vel_y"],
            row["angular_vel_z"],
        ], dtype=float),
    }


def merge_future_target_row(base_row, target_row):
    input_row = base_row.copy()
    for col in FUTURE_TARGET_COLUMNS:
        if col in target_row.index:
            input_row[col] = target_row[col]
    return input_row


def make_input_from_setpoint(row, state, context, dt, horizon_step=None, use_log_feedback=True):
    state = np.asarray(state, dtype=float).copy()
    q = normalize(state[0:4])
    if np.linalg.norm(q) < 1.0e-8:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    state[0:4] = q
    pos_sp, yaw_sp = make_position_setpoint_from_row(row, horizon_step=horizon_step)
    vel = state[[STATE_INDEX["vx"], STATE_INDEX["vy"], STATE_INDEX["vz"]]].copy()
    omega = state[[STATE_INDEX["wx"], STATE_INDEX["wy"], STATE_INDEX["wz"]]].copy()
    if "prev_vel" not in context:
        context["prev_vel"] = vel.copy()
    if "prev_acc" not in context:
        context["prev_acc"] = np.zeros(3, dtype=float)
    if "vel_int" not in context:
        context["vel_int"] = np.zeros(3, dtype=float)
    if "prev_omega" not in context:
        context["prev_omega"] = omega.copy()
    if "prev_omega_dot" not in context:
        context["prev_omega_dot"] = np.zeros(3, dtype=float)
    if "rate_int" not in context:
        logged_rate_int = logged_rate_int_from_row(row)
        context["rate_int"] = (
            logged_rate_int.copy()
            if logged_rate_int is not None
            else np.zeros(3, dtype=float)
        )
    if "yaw_torque_lpf_state" not in context:
        context["yaw_torque_lpf_state"] = None
    if "motor_command_delay_buffer" not in context:
        context["motor_command_delay_buffer"] = []
    if "saturation_positive" not in context:
        context["saturation_positive"] = np.zeros(3, dtype=bool)
    if "saturation_negative" not in context:
        context["saturation_negative"] = np.zeros(3, dtype=bool)
    if "takeoff_state" not in context:
        context["takeoff_state"] = TAKEOFF_STATE_FLIGHT
    if "hover_thrust" not in context:
        context["hover_thrust"] = MPC_THR_HOVER
    if "last_hover_thrust_log" not in context:
        context["last_hover_thrust_log"] = np.nan
    if "acceleration_bias" not in context:
        context["acceleration_bias"] = np.zeros(3, dtype=float)
    if "force_bias" not in context:
        context["force_bias"] = np.zeros(3, dtype=float)
    if "torque_bias" not in context:
        context["torque_bias"] = np.zeros(3, dtype=float)
    if "prev_model_acc" not in context:
        context["prev_model_acc"] = np.full(3, np.nan, dtype=float)
    if "prev_log_vel" not in context:
        context["prev_log_vel"] = np.array([
            row["vel_x"],
            row["vel_y"],
            row["vel_z"],
        ], dtype=float)
    if "next_rate_int_sync_time" not in context:
        if (
            use_log_feedback
            and RATE_INT_SYNC_PERIOD
            and RATE_INT_SYNC_PERIOD > 0.0
            and "control_time_s" in row.index
            and np.isfinite(row["control_time_s"])
        ):
            context["next_rate_int_sync_time"] = (
                float(row["control_time_s"]) + RATE_INT_SYNC_PERIOD
            )
        else:
            context["next_rate_int_sync_time"] = np.inf
    if (
        use_log_feedback
        and RATE_INT_SYNC_PERIOD
        and RATE_INT_SYNC_PERIOD > 0.0
        and "control_time_s" in row.index
        and np.isfinite(row["control_time_s"])
        and float(row["control_time_s"]) + 0.5 * dt >= context["next_rate_int_sync_time"]
    ):
        logged_rate_int = logged_rate_int_from_row(row)
        if logged_rate_int is not None:
            context["rate_int"] = logged_rate_int.copy()
        context["next_rate_int_sync_time"] = (
            float(row["control_time_s"]) + RATE_INT_SYNC_PERIOD
        )

    if use_log_feedback and "hover_thrust" in row.index and "hover_thrust_valid" in row.index:
        hover_thrust_new = float(row["hover_thrust"])
        hover_thrust_valid = bool(
            np.isfinite(row["hover_thrust_valid"])
            and row["hover_thrust_valid"] != 0.0
        )
        hover_thrust_updated = (
            not np.isfinite(context["last_hover_thrust_log"])
            or abs(hover_thrust_new - context["last_hover_thrust_log"]) > 1.0e-7
        )
        if (
            hover_thrust_valid
            and hover_thrust_updated
            and np.isfinite(hover_thrust_new)
            and hover_thrust_new > 1.0e-6
        ):
            context["hover_thrust"] = hover_thrust_new
            context["last_hover_thrust_log"] = hover_thrust_new

    landed = bool(row["landed"]) if "landed" in row.index and np.isfinite(row["landed"]) else False
    maybe_landed = (
        bool(row["maybe_landed"])
        if "maybe_landed" in row.index and np.isfinite(row["maybe_landed"])
        else False
    )
    ground_contact = (
        bool(row["ground_contact"])
        if "ground_contact" in row.index and np.isfinite(row["ground_contact"])
        else False
    )
    if "takeoff_state" in row.index and np.isfinite(row["takeoff_state"]):
        context["takeoff_state"] = int(row["takeoff_state"])
    tilt_limit = (
        float(row["takeoff_tilt_limit"])
        if "takeoff_tilt_limit" in row.index and np.isfinite(row["takeoff_tilt_limit"])
        else MPC_TILT_MAX
    )
    takeoff_state = context["takeoff_state"]
    not_taken_off = takeoff_state < TAKEOFF_STATE_RAMPUP
    flying = takeoff_state >= TAKEOFF_STATE_FLIGHT
    flying_but_ground_contact = flying and (ground_contact or maybe_landed)
    no_thrust = bool(not_taken_off or flying_but_ground_contact)
    thrust_min = MPC_THR_MIN if flying else 0.0
    z_vel_max_up = MPC_Z_VEL_MAX_UP

    current_log_vel = np.array([row["vel_x"], row["vel_y"], row["vel_z"]], dtype=float)
    acceleration_bias_observed = context["acceleration_bias"].copy()
    if (
        use_log_feedback
        and USE_ACCELERATION_BIAS_OBSERVER
        and USE_VELOCITY_DELTA_BIAS_OBSERVER
        and np.any(VELOCITY_DELTA_BIAS_AXES)
        and np.all(np.isfinite(current_log_vel))
        and np.all(np.isfinite(context["prev_log_vel"]))
        and np.all(np.isfinite(context["prev_model_acc"]))
    ):
        logged_acc = (current_log_vel - context["prev_log_vel"]) / dt
        residual = logged_acc - context["prev_model_acc"]
        residual[np.abs(residual) < ACCELERATION_BIAS_DEADBAND] = 0.0
        axes = VELOCITY_DELTA_BIAS_AXES
        context["acceleration_bias"][axes] = (
            context["acceleration_bias"][axes]
            + VELOCITY_DELTA_BIAS_ALPHA * residual[axes]
        )
        context["acceleration_bias"] = np.clip(
            context["acceleration_bias"],
            -ACCELERATION_BIAS_LIMIT,
            ACCELERATION_BIAS_LIMIT,
        )
        acceleration_bias_observed = context["acceleration_bias"].copy()

    if use_log_feedback:
        logged_vel_int = estimate_vel_int_from_logged_setpoint(row, state, context["prev_acc"])
        if logged_vel_int is not None:
            context["vel_int"] = logged_vel_int
    prev_vel = context["prev_vel"]
    prev_acc = context["prev_acc"]
    vel_int = context["vel_int"]
    prev_omega = context["prev_omega"]
    prev_omega_dot = context["prev_omega_dot"]
    rate_int = context["rate_int"]
    hover_thrust = context["hover_thrust"]
    acceleration_bias = context["acceleration_bias"].copy()
    force_bias = context["force_bias"].copy()
    torque_bias = context["torque_bias"].copy()

    if no_thrust:
        if use_log_feedback:
            context["prev_log_vel"] = current_log_vel.copy()
        input_data = {"kind": "hold"}
        debug = {
            "pos_sp": pos_sp,
            "yaw_sp": np.array([yaw_sp], dtype=float),
            "vel_sp": np.full(3, np.nan),
            "acc_sp": np.array([0.0, 0.0, 100.0], dtype=float),
            "thrust_sp": np.zeros(3, dtype=float),
            "thrust_sp_ned": np.zeros(3, dtype=float),
            "att_sp": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            "rate_sp": np.zeros(3, dtype=float),
            "torque_sp": np.zeros(3, dtype=float),
            "torque_sp_unfiltered": np.zeros(3, dtype=float),
            "motor_setpoint": np.zeros(4, dtype=float),
            "motor_setpoint_used": np.zeros(4, dtype=float),
            "allocator_unallocated": np.zeros(4, dtype=float),
            "acceleration_bias": acceleration_bias,
            "force_bias": force_bias,
            "torque_bias": torque_bias,
            "acceleration_bias_observed": acceleration_bias_observed,
            "takeoff_state": np.array([takeoff_state], dtype=float),
            "thrust_min": np.array([thrust_min], dtype=float),
            "tilt_limit": np.array([tilt_limit], dtype=float),
            "no_thrust": np.array([1.0], dtype=float),
        }
        return input_data, debug, context

    vel_dot = (vel - prev_vel) / dt
    if MPC_VELD_LP > 1.0e-6:
        vel_dot_alpha = dt / (dt + 1.0 / (2.0 * np.pi * MPC_VELD_LP))
    else:
        vel_dot_alpha = 1.0
    vel_dot_lpf = prev_acc + vel_dot_alpha * (vel_dot - prev_acc)
    vel_sp = np.array([
        MPC_XY_P * (pos_sp[0] - state[STATE_INDEX["x"]]),
        MPC_XY_P * (pos_sp[1] - state[STATE_INDEX["y"]]),
        MPC_Z_P * (pos_sp[2] - state[STATE_INDEX["z"]]),
    ], dtype=float)
    vel_xy_norm = np.linalg.norm(vel_sp[:2])
    if vel_xy_norm > MPC_XY_VEL_MAX and vel_xy_norm > 1.0e-8:
        vel_sp[:2] = vel_sp[:2] / vel_xy_norm * MPC_XY_VEL_MAX
    vel_sp[2] = np.clip(vel_sp[2], -z_vel_max_up, MPC_Z_VEL_MAX_DOWN)
    vel_error = vel_sp - vel
    acc_sp = np.array([
        MPC_XY_VEL_P_ACC * vel_error[0] + vel_int[0] - MPC_XY_VEL_D_ACC * vel_dot_lpf[0],
        MPC_XY_VEL_P_ACC * vel_error[1] + vel_int[1] - MPC_XY_VEL_D_ACC * vel_dot_lpf[1],
        MPC_Z_VEL_P_ACC * vel_error[2] + vel_int[2] - MPC_Z_VEL_D_ACC * vel_dot_lpf[2],
    ], dtype=float)
    z_specific_force = A_OF_GRAVITY if MPC_ACC_DECOUPLE else A_OF_GRAVITY - acc_sp[2]
    body_z = normalize(np.array([-acc_sp[0], -acc_sp[1], z_specific_force], dtype=float))
    if np.linalg.norm(body_z) < 1.0e-8:
        body_z = np.array([0.0, 0.0, 1.0], dtype=float)
    tilt_angle = np.arccos(np.clip(body_z[2], -1.0, 1.0))
    if tilt_angle > tilt_limit:
        rejection = np.array([body_z[0], body_z[1], 0.0], dtype=float)
        rejection_norm = np.linalg.norm(rejection)
        if rejection_norm < 1.0e-8:
            rejection = np.array([1.0, 0.0, 0.0], dtype=float)
            rejection_norm = 1.0
        rejection = rejection / rejection_norm
        body_z = np.array([
            np.sin(tilt_limit) * rejection[0],
            np.sin(tilt_limit) * rejection[1],
            np.cos(tilt_limit),
        ], dtype=float)
    thrust_ned_z = acc_sp[2] * (hover_thrust / A_OF_GRAVITY) - hover_thrust
    cos_ned_body = body_z[2] if abs(body_z[2]) >= 1.0e-6 else 1.0e-6
    collective_thrust = min(thrust_ned_z / cos_ned_body, -thrust_min)
    thrust_sp_ned = body_z * collective_thrust
    vel_error_for_int = vel_error.copy()
    if (
        thrust_sp_ned[2] >= -thrust_min and vel_error_for_int[2] >= 0.0
    ) or (
        thrust_sp_ned[2] <= -MPC_THR_MAX and vel_error_for_int[2] <= 0.0
    ):
        vel_error_for_int[2] = 0.0
    thrust_sp_xy_norm = np.linalg.norm(thrust_sp_ned[:2])
    thrust_max_squared = MPC_THR_MAX * MPC_THR_MAX
    allocated_horizontal_thrust = min(thrust_sp_xy_norm, MPC_THR_XY_MARGIN)
    thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust ** 2
    thrust_sp_ned[2] = max(thrust_sp_ned[2], -np.sqrt(max(0.0, thrust_z_max_squared)))
    thrust_max_xy_squared = thrust_max_squared - thrust_sp_ned[2] ** 2
    thrust_max_xy = np.sqrt(max(0.0, thrust_max_xy_squared))
    if thrust_sp_xy_norm > thrust_max_xy and thrust_sp_xy_norm > 1.0e-8:
        thrust_sp_ned[:2] = thrust_sp_ned[:2] / thrust_sp_xy_norm * thrust_max_xy

    acc_sp_xy_produced = thrust_sp_ned[:2] * (A_OF_GRAVITY / hover_thrust)
    if np.dot(acc_sp[:2], acc_sp[:2]) > np.dot(acc_sp_xy_produced, acc_sp_xy_produced):
        arw_gain = 2.0 / MPC_XY_VEL_P_ACC
        vel_error_for_int[:2] -= arw_gain * (acc_sp[:2] - acc_sp_xy_produced)
    vel_error_for_int[~np.isfinite(vel_error_for_int)] = 0.0

    thrust_sp_body = np.array([0.0, 0.0, -np.linalg.norm(thrust_sp_ned)], dtype=float)
    att_body_z = normalize(-thrust_sp_ned)
    if np.linalg.norm(att_body_z) < 1.0e-8:
        att_body_z = np.array([0.0, 0.0, 1.0], dtype=float)
    sy, cy = sin_cos_simulator(float(yaw_sp))
    y_c = np.array([-sy, cy, 0.0], dtype=float)
    body_x = np.cross(y_c, att_body_z)
    if att_body_z[2] < 0.0:
        body_x = -body_x
    if abs(att_body_z[2]) < 1.0e-6:
        body_x = np.array([0.0, 0.0, 1.0], dtype=float)
    body_x = normalize(body_x)
    body_y = np.cross(att_body_z, body_x)
    att_sp = rotmat_to_quat(body_x, body_y, att_body_z)

    qd = att_sp.copy()
    e_z = quat_to_rotmat(q)[:, 2]
    e_z_d = quat_to_rotmat(qd)[:, 2]
    tilt_axis = np.cross(e_z, e_z_d)
    tilt_axis_norm = np.linalg.norm(tilt_axis)
    tilt_dot = np.clip(np.dot(e_z, e_z_d), -1.0, 1.0)
    if tilt_axis_norm < 1.0e-8:
        qd_red = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if tilt_dot > 0.0 else qd.copy()
    else:
        tilt_axis /= tilt_axis_norm
        tilt_angle = np.arctan2(tilt_axis_norm, tilt_dot)
        qd_red = normalize(np.array([
            np.cos(0.5 * tilt_angle),
            tilt_axis[0] * np.sin(0.5 * tilt_angle),
            tilt_axis[1] * np.sin(0.5 * tilt_angle),
            tilt_axis[2] * np.sin(0.5 * tilt_angle),
        ], dtype=float))
        if abs(qd_red[1]) > 1.0 - 1.0e-5 or abs(qd_red[2]) > 1.0 - 1.0e-5:
            qd_red = qd.copy()
        else:
            qd_red = normalize(quat_multiply(qd_red, q))
    qd_red_inv = np.array([qd_red[0], -qd_red[1], -qd_red[2], -qd_red[3]], dtype=float)
    qd_dyaw = normalize(quat_multiply(qd_red_inv, qd))
    if qd_dyaw[0] < 0.0:
        qd_dyaw = -qd_dyaw
    qd_dyaw[0] = np.clip(qd_dyaw[0], -1.0, 1.0)
    qd_dyaw[3] = np.clip(qd_dyaw[3], -1.0, 1.0)
    q_yaw_weighted = np.array([
        np.cos(MC_YAW_WEIGHT * np.arccos(qd_dyaw[0])),
        0.0,
        0.0,
        np.sin(MC_YAW_WEIGHT * np.arcsin(qd_dyaw[3])),
    ], dtype=float)
    qd_weighted = normalize(quat_multiply(qd_red, q_yaw_weighted))
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
    qe = normalize(quat_multiply(q_inv, qd_weighted))
    if qe[0] < 0.0:
        qe = -qe
    attitude_gain = np.array([
        MC_ROLL_P,
        MC_PITCH_P,
        MC_YAW_P / MC_YAW_WEIGHT if MC_YAW_WEIGHT > 1.0e-4 else MC_YAW_P,
    ], dtype=float)
    rate_sp = 2.0 * qe[1:4] * attitude_gain
    rate_sp = np.clip(
        rate_sp,
        -np.array([MC_ROLLRATE_MAX, MC_PITCHRATE_MAX, MC_YAWRATE_MAX]),
        np.array([MC_ROLLRATE_MAX, MC_PITCHRATE_MAX, MC_YAWRATE_MAX]),
    )
    raw_omega_dot = (omega - prev_omega) / dt
    if ANGULAR_ACCEL_LP > 1.0e-6:
        omega_dot_alpha = dt / (dt + 1.0 / (2.0 * np.pi * ANGULAR_ACCEL_LP))
        omega_dot = prev_omega_dot + omega_dot_alpha * (raw_omega_dot - prev_omega_dot)
    else:
        omega_dot = raw_omega_dot
    rate_error = rate_sp - omega
    rate_p = np.array([
        MC_ROLLRATE_K * MC_ROLLRATE_P,
        MC_PITCHRATE_K * MC_PITCHRATE_P,
        MC_YAWRATE_K * MC_YAWRATE_P,
    ], dtype=float)
    rate_i = np.array([
        MC_ROLLRATE_K * MC_ROLLRATE_I,
        MC_PITCHRATE_K * MC_PITCHRATE_I,
        MC_YAWRATE_K * MC_YAWRATE_I,
    ], dtype=float)
    rate_d = np.array([
        MC_ROLLRATE_K * MC_ROLLRATE_D,
        MC_PITCHRATE_K * MC_PITCHRATE_D,
        MC_YAWRATE_K * MC_YAWRATE_D,
    ], dtype=float)
    rate_ff = np.array([MC_ROLLRATE_FF, MC_PITCHRATE_FF, MC_YAWRATE_FF], dtype=float)
    rate_int_lim = np.array([MC_RR_INT_LIM, MC_PR_INT_LIM, MC_YR_INT_LIM], dtype=float)
    torque_p = rate_p * rate_error
    torque_i = rate_int.copy()
    torque_d = -rate_d * omega_dot
    torque_ff = rate_ff * rate_sp
    torque_sp_unfiltered = torque_p + torque_i + torque_d + torque_ff
    torque_sp = torque_sp_unfiltered.copy()
    if MC_YAW_TQ_CUTOFF > 1.0e-6:
        yaw_alpha = dt / (dt + 1.0 / (2.0 * np.pi * MC_YAW_TQ_CUTOFF))
        yaw_prev = (
            torque_sp[2]
            if context["yaw_torque_lpf_state"] is None
            else float(context["yaw_torque_lpf_state"])
        )
        torque_sp[2] = yaw_prev + yaw_alpha * (torque_sp[2] - yaw_prev)
        yaw_torque_lpf_next = torque_sp[2]
    else:
        yaw_torque_lpf_next = torque_sp[2]
    rate_int_next = rate_int.copy()
    saturation_positive = np.asarray(context["saturation_positive"], dtype=bool)
    saturation_negative = np.asarray(context["saturation_negative"], dtype=bool)
    for axis in range(3):
        rate_error_for_int = rate_error[axis]
        if saturation_positive[axis]:
            rate_error_for_int = min(rate_error_for_int, 0.0)
        if saturation_negative[axis]:
            rate_error_for_int = max(rate_error_for_int, 0.0)
        i_factor = rate_error_for_int / np.deg2rad(400.0)
        i_factor = max(0.0, 1.0 - i_factor * i_factor)
        rate_int_next[axis] += i_factor * rate_i[axis] * rate_error_for_int * dt
        if np.isfinite(rate_int_next[axis]):
            rate_int_next[axis] = np.clip(rate_int_next[axis], -rate_int_lim[axis], rate_int_lim[axis])
        else:
            rate_int_next[axis] = rate_int[axis]
    control_sp = np.array([
        torque_sp[0],
        torque_sp[1],
        torque_sp[2],
        thrust_sp_body[2],
    ], dtype=float)
    motor_setpoint, unallocated = allocate_px4_quad_x(control_sp)
    motor_setpoint_used = apply_motor_command_delay(
        motor_setpoint,
        context["motor_command_delay_buffer"],
        MOTOR_COMMAND_DELAY_STEPS,
    )
    bias_torque = (
        estimate_rate_int_bias_torque(rate_int, thrust_sp_body[2], dt)
        if should_apply_rate_int_bias_torque(False)
        else np.zeros(3, dtype=float)
    )
    vel_int_next = vel_int.copy()
    vel_int_next[0] += vel_error_for_int[0] * MPC_XY_VEL_I_ACC * dt
    vel_int_next[1] += vel_error_for_int[1] * MPC_XY_VEL_I_ACC * dt
    vel_int_next[2] += vel_error_for_int[2] * MPC_Z_VEL_I_ACC * dt
    vel_int_next[2] = np.clip(vel_int_next[2], -A_OF_GRAVITY, A_OF_GRAVITY)
    context["prev_vel"] = vel.copy()
    context["prev_acc"] = vel_dot_lpf.copy()
    context["vel_int"] = vel_int_next
    context["prev_omega"] = omega.copy()
    context["prev_omega_dot"] = omega_dot.copy()
    context["rate_int"] = rate_int_next
    context["yaw_torque_lpf_state"] = yaw_torque_lpf_next
    context["saturation_positive"] = unallocated[:3] > np.finfo(float).eps
    context["saturation_negative"] = unallocated[:3] < -np.finfo(float).eps
    input_data = {
        "kind": "actuator",
        "actuator": motor_setpoint_used,
        "bias_torque": bias_torque,
        "acceleration_bias": acceleration_bias,
        "force_bias": force_bias,
        "torque_bias": torque_bias,
    }
    debug = {
        "pos_sp": pos_sp,
        "yaw_sp": np.array([yaw_sp], dtype=float),
        "vel_sp": vel_sp,
        "acc_sp": acc_sp,
        "thrust_sp": thrust_sp_body,
        "thrust_sp_ned": thrust_sp_ned,
        "att_sp": att_sp,
        "rate_sp": rate_sp,
        "rate_derivative": omega_dot,
        "rate_int": rate_int,
        "rate_int_next": rate_int_next,
        "torque_sp": torque_sp,
        "torque_sp_unfiltered": torque_sp_unfiltered,
        "torque_p": torque_p,
        "torque_i": torque_i,
        "torque_d": torque_d,
        "torque_ff": torque_ff,
        "motor_setpoint": motor_setpoint,
        "motor_setpoint_used": motor_setpoint_used,
        "allocator_unallocated": unallocated,
        "bias_torque": bias_torque,
        "acceleration_bias": acceleration_bias,
        "force_bias": force_bias,
        "torque_bias": torque_bias,
        "acceleration_bias_observed": acceleration_bias_observed,
        "takeoff_state": np.array([takeoff_state], dtype=float),
        "thrust_min": np.array([thrust_min], dtype=float),
        "tilt_limit": np.array([tilt_limit], dtype=float),
        "no_thrust": np.array([0.0], dtype=float),
        "hover_thrust": np.array([hover_thrust], dtype=float),
        "vel_int": vel_int,
        "vel_int_next": vel_int_next,
    }
    if use_log_feedback:
        context["prev_log_vel"] = current_log_vel.copy()
    return input_data, debug, context
def warm_prediction_context(df, start_idx, dt, mode=MODE_MODEL_INPUT):
    context = {}
    prev_motor_speed = None
    warm_end = int(np.clip(start_idx, 0, len(df)))
    for row_idx in range(warm_end):
        if (
            row_idx > 0
            and "_segment_id" in df.columns
            and df.loc[row_idx, "_segment_id"] != df.loc[row_idx - 1, "_segment_id"]
        ):
            context = {}
            prev_motor_speed = None
        row = df.loc[row_idx]
        state = state_from_row(row)
        input_data, input_debug, context = make_input_from_row(row, state, mode, context, dt)
        _, debug = dynamics_step(
            state,
            input_data,
            dt,
            prev_motor_speed=prev_motor_speed,
        )
        debug.update(input_debug)
        debug.update(
            update_dynamics_bias_observer(
                context,
                row,
                next_row_for_observer(df, row_idx),
                state,
                debug,
                dt,
            )
        )
        if "acc_used" in debug:
            context["prev_model_acc"] = np.asarray(debug["acc_used"], dtype=float).copy()
        prev_motor_speed = debug.get("motor_speed", prev_motor_speed)
    return context, prev_motor_speed


def build_prediction_contexts(df, start_indices, dt, mode=MODE_MODEL_INPUT):
    start_set = set(int(idx) for idx in start_indices)
    if not start_set:
        return {}
    max_start = max(start_set)
    context = {}
    prev_motor_speed = None
    contexts = {}
    for row_idx in range(min(max_start + 1, len(df))):
        if (
            row_idx > 0
            and "_segment_id" in df.columns
            and df.loc[row_idx, "_segment_id"] != df.loc[row_idx - 1, "_segment_id"]
        ):
            context = {}
            prev_motor_speed = None
        if row_idx in start_set:
            contexts[row_idx] = (
                copy.deepcopy(context),
                None if prev_motor_speed is None else prev_motor_speed.copy(),
            )
        row = df.loc[row_idx]
        state = state_from_row(row)
        input_data, input_debug, context = make_input_from_row(row, state, mode, context, dt)
        _, debug = dynamics_step(
            state,
            input_data,
            dt,
            prev_motor_speed=prev_motor_speed,
        )
        debug.update(input_debug)
        debug.update(
            update_dynamics_bias_observer(
                context,
                row,
                next_row_for_observer(df, row_idx),
                state,
                debug,
                dt,
            )
        )
        if "acc_used" in debug:
            context["prev_model_acc"] = np.asarray(debug["acc_used"], dtype=float).copy()
        prev_motor_speed = debug.get("motor_speed", prev_motor_speed)
    return contexts


def rollout(
    df,
    start_idx,
    horizon,
    dt,
    mode=MODE_ACTUATOR_LOG,
    initial_context=None,
    initial_prev_motor_speed=None,
):
    state = state_from_row(df.loc[start_idx])
    pred_states = np.full((horizon + 1, len(STATE_NAMES)), np.nan)
    pred_states[0] = state.copy()
    prev_motor_speed = initial_prev_motor_speed
    context = copy.deepcopy(initial_context) if initial_context is not None else {}
    debug_list = []
    start_segment = df.loc[start_idx, "_segment_id"] if "_segment_id" in df.columns else None
    realtime_input_row = (
        df.loc[start_idx].copy()
        if mode in (MODE_MODEL_INPUT, MODE_ACTUATOR_LOG)
        else None
    )
    for h in range(horizon):
        row_idx = start_idx + h
        if row_idx >= len(df):
            break
        if start_segment is not None and df.loc[row_idx, "_segment_id"] != start_segment:
            break
        row = df.loc[row_idx]
        if mode == MODE_MODEL_INPUT:
            target_input_row = merge_future_target_row(realtime_input_row, row)
            input_data, input_debug, context = make_input_from_setpoint(
                target_input_row,
                state,
                context,
                dt,
                horizon_step=h,
                use_log_feedback=False,
            )
        elif mode == MODE_ACTUATOR_LOG:
            actuator = actuator_motor_from_row(row)
            if actuator is None:
                raise ValueError("actuator_motor input is missing or invalid")
            input_data, input_debug, context = make_input_from_setpoint(
                realtime_input_row,
                state,
                context,
                dt,
                horizon_step=h,
                use_log_feedback=False,
            )
            input_debug["calculated_motor_setpoint_used"] = np.asarray(
                input_data["actuator"],
                dtype=float,
            ).copy()
            input_data["actuator"] = actuator
            input_debug["motor_setpoint_used"] = actuator.copy()
        else:
            input_data, input_debug, context = make_input_from_row(row, state, mode, context, dt)
        state, debug = dynamics_step(
            state,
            input_data,
            dt,
            prev_motor_speed=prev_motor_speed,
        )
        debug.update(input_debug)
        if "acc_used" in debug:
            context["prev_model_acc"] = np.asarray(debug["acc_used"], dtype=float).copy()
        prev_motor_speed = debug.get("motor_speed", prev_motor_speed)
        pred_states[h + 1] = state.copy()
        debug_list.append(debug)
    return pred_states, debug_list

def state_series_from_df(df, state_name):
    if state_name in ("roll", "pitch", "yaw"):
        if state_name not in df.columns:
            raise ValueError(f"missing log column: {state_name}")
        return df[state_name].to_numpy(float)
    log_columns = {
        "x": "pos_x",
        "y": "pos_y",
        "z": "pos_z",
        "vx": "vel_x",
        "vy": "vel_y",
        "vz": "vel_z",
        "wx": "angular_vel_x",
        "wy": "angular_vel_y",
        "wz": "angular_vel_z",
    }
    if state_name in log_columns:
        col = log_columns[state_name]
        if col not in df.columns:
            raise ValueError(f"missing log column: {col}")
        return df[col].to_numpy(float)
    if state_name in STATE_INDEX:
        return np.array([state_from_row(row)[STATE_INDEX[state_name]] for _, row in df.iterrows()])
    raise ValueError(f"unknown state: {state_name}")

def state_series_from_states(states, state_name):
    states = np.asarray(states, dtype=float)
    if state_name in ("roll", "pitch", "yaw"):
        euler = np.array([quat_to_euler(state[0:4]) for state in states])
        return euler[:, {"roll": 0, "pitch": 1, "yaw": 2}[state_name]]

    if state_name in STATE_INDEX:
        return states[:, STATE_INDEX[state_name]]

    raise ValueError(f"unknown state: {state_name}")


def time_array_from_df(df, dt=DT):
    if "_segment_time_s" in df.columns:
        t = df["_segment_time_s"].to_numpy(float)
        if np.any(np.isfinite(t)):
            return t
    if "control_time_s" in df.columns:
        t = df["control_time_s"].to_numpy(float)
        if np.any(np.isfinite(t)):
            finite_t = t[np.isfinite(t)]
            return t - finite_t[0]
    return np.arange(len(df), dtype=float) * dt


def index_from_time(plot_t, start_time):
    if start_time is None:
        return None
    plot_t = np.asarray(plot_t, dtype=float)
    finite = np.flatnonzero(np.isfinite(plot_t))
    if len(finite) == 0:
        return 0
    return int(finite[np.argmin(np.abs(plot_t[finite] - float(start_time)))])


def default_display_mask(df):
    if "_segment_id" not in df.columns:
        return np.ones(len(df), dtype=bool)
    segment_id = df["_segment_id"].to_numpy()
    finite = segment_id[np.isfinite(segment_id)]
    if len(finite) == 0:
        return np.ones(len(df), dtype=bool)
    return segment_id == np.max(finite)


def prediction_start_indices(plot_t, display_mask, prediction_interval=PREDICTION_INTERVAL):
    indices = np.flatnonzero(display_mask & np.isfinite(plot_t))
    if len(indices) == 0:
        return np.array([], dtype=int)
    if prediction_interval <= 0.0:
        return indices[:1]
    starts = [int(indices[0])]
    next_time = float(plot_t[indices[0]]) + prediction_interval
    for idx in indices[1:]:
        if float(plot_t[idx]) + 1.0e-9 >= next_time:
            starts.append(int(idx))
            next_time = float(plot_t[idx]) + prediction_interval
    return np.asarray(starts, dtype=int)


def plot_state(
    df,
    state_name,
    start_idx=0,
    start_time=None,
    horizon=PREDICTION_HORIZON,
    dt=DT,
    mode=MODE_MODEL_INPUT,
    prediction_interval=PREDICTION_INTERVAL,
    plot_start=None,
    plot_end=None,
    auto_window=False,
    window_before=1.0,
    window_after=None,
    display_last_segment=True,
    ax=None,
    show=True,
):
    plot_t = time_array_from_df(df, dt)
    horizon = int(max(1, horizon))
    actual_value = state_series_from_df(df, state_name)
    display_mask = (
        default_display_mask(df)
        if display_last_segment
        else np.ones(len(df), dtype=bool)
    )

    time_start_idx = index_from_time(plot_t, start_time)
    if time_start_idx is not None:
        start_indices = np.array([time_start_idx], dtype=int)
    else:
        start_indices = prediction_start_indices(
            plot_t,
            display_mask,
            prediction_interval=prediction_interval,
        )
        if len(start_indices) == 0:
            start_idx = int(np.clip(start_idx, 0, max(len(df) - 1, 0)))
            start_indices = np.array([start_idx], dtype=int)

    if ax is None:
        _, ax = plt.subplots()
    actual_mask = display_mask & np.isfinite(plot_t) & np.isfinite(actual_value)
    if auto_window and plot_start is None and plot_end is None:
        if window_after is None:
            window_after = horizon * dt + 1.0
        first_start = int(start_indices[0])
        plot_start = plot_t[first_start] - window_before
        plot_end = plot_t[first_start] + window_after
    if plot_start is not None:
        actual_mask &= plot_t >= float(plot_start) - 1.0e-9
    if plot_end is not None:
        actual_mask &= plot_t <= float(plot_end) + 1.0e-9
    ax.plot(
        plot_t[actual_mask],
        actual_value[actual_mask],
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label=f"{state_name}_actual",
    )
    pred_rollouts = []
    context_by_start = build_prediction_contexts(df, start_indices, dt, mode=mode)
    for pred_i, pred_start_idx in enumerate(start_indices):
        pred_start_idx = int(np.clip(pred_start_idx, 0, max(len(df) - 1, 0)))
        initial_context, initial_prev_motor_speed = context_by_start.get(
            pred_start_idx,
            ({}, None),
        )
        pred_states, _ = rollout(
            df,
            pred_start_idx,
            horizon,
            dt,
            mode=mode,
            initial_context=initial_context,
            initial_prev_motor_speed=initial_prev_motor_speed,
        )
        pred_rollouts.append((pred_start_idx, pred_states))
        pred_value = state_series_from_states(pred_states, state_name)
        pred_t = plot_t[pred_start_idx] + np.arange(len(pred_value), dtype=float) * dt
        pred_mask = np.isfinite(pred_t) & np.isfinite(pred_value)
        if plot_start is not None:
            pred_mask &= pred_t >= float(plot_start) - 1.0e-9
        if plot_end is not None:
            pred_mask &= pred_t <= float(plot_end) + 1.0e-9
        ax.plot(
            pred_t[pred_mask],
            pred_value[pred_mask],
            color="tab:cyan",
            linewidth=2.0,
            alpha=0.8,
            label=f"{state_name}_{mode}" if pred_i == 0 else None,
        )
    if len(start_indices) > 0:
        ax.axvline(plot_t[int(start_indices[0])], color="0.4", linewidth=1.0, alpha=0.5)
    ax.set_title(state_name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(state_name)
    ax.grid(True)
    ax.legend()
    if plot_start is not None or plot_end is not None or np.any(actual_mask):
        left = float(plot_start) if plot_start is not None else np.nanmin(plot_t[actual_mask])
        right = float(plot_end) if plot_end is not None else np.nanmax(plot_t[actual_mask])
        if np.isfinite(left) and np.isfinite(right) and right > left:
            ax.set_xlim(left, right)
    if show:
        plt.show()
    return ax, pred_rollouts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--state", default="x")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON)
    parser.add_argument("--prediction-interval", type=float, default=PREDICTION_INTERVAL)
    parser.add_argument("--plot-start", type=float, default=None)
    parser.add_argument("--plot-end", type=float, default=None)
    parser.add_argument("--full-range", action="store_true")
    parser.add_argument("--auto-window", action="store_true")
    parser.add_argument("--all-segments", action="store_true")
    parser.add_argument(
        "--mode",
        default=MODE_MODEL_INPUT,
        choices=[MODE_MODEL_INPUT, MODE_ACTUATOR_LOG, MODE_ACTUAL_STATE],
    )
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    df = preprocess_log(df, DT)
    df = add_actuator_input_columns(df)
    plot_state(
        df,
        args.state,
        start_idx=args.start_idx,
        start_time=args.start_time,
        horizon=args.horizon,
        dt=DT,
        mode=args.mode,
        prediction_interval=args.prediction_interval,
        plot_start=args.plot_start,
        plot_end=args.plot_end,
        auto_window=args.auto_window and not args.full_range,
        display_last_segment=not args.all_segments,
        show=True,
    )
if __name__ == "__main__":
    main()
