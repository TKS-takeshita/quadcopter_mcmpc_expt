import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_20260528_213107.csv"

plot_name = sys.argv[1] if len(sys.argv) > 1 else "vx"
csv_path = sys.argv[2] if len(sys.argv) > 2 else default_csv

dt = 0.02

# ===== PX4/MPC parameters =====
A_OF_GRAVITY = 9.80665
MAX_THRUST = 38.42

MPC_XY_P = 0.30
MPC_Z_P = 1.00

MPC_XY_VEL_P_ACC = 1.80
MPC_XY_VEL_I_ACC = 0.00
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

        pos_sp = np.array([
            row["u0_x"],
            row["u0_y"],
            row["u0_z"],
        ])

        yaw_sp = row["u0_yaw"]

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

        # ==============================
        # PX4 _velocityControl()
        # acc_sp = P * vel_error + vel_int - D * vel_dot
        # ==============================
        vel_error = vel_sp - vel

        # ===== modified: raw derivative -> LPF derivative =====
        if i == 0:
            vel_dot = np.zeros(3)
            vel_dot_lpf[:] = 0.0
        else:
            vel_dot_raw = (vel - prev_vel) / dt
            vel_dot_lpf = vel_dot_lpf + vel_dot_alpha * (vel_dot_raw - vel_dot_lpf)
            vel_dot = vel_dot_lpf

        acc_sp = np.zeros(3)

        acc_sp[0] = (
            MPC_XY_VEL_P_ACC * vel_error[0]
            + vel_int[0]
            - MPC_XY_VEL_D_ACC * vel_dot[0]
        )

        acc_sp[1] = (
            MPC_XY_VEL_P_ACC * vel_error[1]
            + vel_int[1]
            - MPC_XY_VEL_D_ACC * vel_dot[1]
        )

        acc_sp[2] = (
            MPC_Z_VEL_P_ACC * vel_error[2]
            + vel_int[2]
            - MPC_Z_VEL_D_ACC * vel_dot[2]
        )

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
        att_sp = thrust_to_attitude(thr_sp, yaw_sp)

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

        prev_vel = vel.copy()

    return calc


df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df = df.reset_index(drop=True)

if "t" not in df.columns:
    df["t"] = np.arange(len(df)) * dt

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

fig, ax = plt.subplots(figsize=(12, 5))
plt.subplots_adjust(bottom=0.25, right=0.88)

plot_delay_sec = 0.1 #0.19

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