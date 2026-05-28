import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection

default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_20260528_023838.csv"

state = sys.argv[1] if len(sys.argv) > 1 else "x"
csv_path = sys.argv[2] if len(sys.argv) > 2 else default_csv

simulation = True

dt = 0.02
horizon = 50

# 予測線を描き始める時刻 [s]
prediction_start_time = 0.5

# 予測線を描く間隔 [s]
prediction_interval = 1.0

A_OF_GRAVITY = 9.80665
MAX_THRUST = 38.42

if simulation:
    MPC_XY_P = 0.30
    MPC_Z_P = 1.00
    MPC_XY_VEL_P_ACC = 1.80
    MPC_XY_VEL_I_ACC = 0.40
    MPC_XY_VEL_D_ACC = 0.20
    MPC_Z_VEL_P_ACC = 4.00
    MPC_VELD_LP = 5.0
else:
    MPC_XY_P = 0.95
    MPC_Z_P = 1.00
    MPC_XY_VEL_P_ACC = 1.80
    MPC_XY_VEL_I_ACC = 0.40
    MPC_XY_VEL_D_ACC = 0.20
    MPC_Z_VEL_P_ACC = 4.00
    MPC_VELD_LP = 5.0

LPF = (2.0 * np.pi * MPC_VELD_LP) / (1.0 + 2.0 * np.pi * MPC_VELD_LP)

state_names = [
    "e0", "e1", "e2", "e3",
    "wx", "wy", "wz",
    "x", "y", "z",
    "vx", "vy", "vz",
]

state_index = {name: i for i, name in enumerate(state_names)}

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df = df.reset_index(drop=True)

df["t"] = np.arange(len(df)) * dt

cur_col = f"cur_{state}"
target_col = f"target_{state}"
u0_col = f"u0_{state}"

if cur_col not in df.columns:
    print(f"[ERROR] missing column: {cur_col}")
    sys.exit(1)


def calc_prediction_one_row(row_idx):
    row = df.iloc[row_idx]

    s = np.zeros(len(state_names))

    # 予測の初期状態は必ず現在状態 cur_* から開始
    for name in state_names:
        s[state_index[name]] = row[f"cur_{name}"]

    if row_idx == 0:
        prev_vel = np.array([
            row["cur_vx"],
            row["cur_vy"],
            row["cur_vz"],
        ])
        prev_acc = np.zeros(3)
    else:
        prev_row = df.iloc[row_idx - 1]

        prev_vel = np.array([
            prev_row["cur_vx"],
            prev_row["cur_vy"],
            prev_row["cur_vz"],
        ])

        cur_vel = np.array([
            row["cur_vx"],
            row["cur_vy"],
            row["cur_vz"],
        ])

        prev_acc = LPF * ((cur_vel - prev_vel) / dt)

    vel_int = np.zeros(3)

    pred = np.zeros((horizon + 1, len(state_names)))
    pred[0] = s.copy()

    for h in range(horizon):
        x_ref = row[f"u{h}_x"]
        y_ref = row[f"u{h}_y"]
        z_ref = row[f"u{h}_z"]

        x_now = s[state_index["x"]]
        y_now = s[state_index["y"]]
        z_now = s[state_index["z"]]

        vx = s[state_index["vx"]]
        vy = s[state_index["vy"]]
        vz = s[state_index["vz"]]

        vel_sp = np.zeros(3)
        vel_sp[0] = MPC_XY_P * (x_ref - x_now)
        vel_sp[1] = MPC_XY_P * (y_ref - y_now)
        vel_sp[2] = MPC_Z_P * (z_ref - z_now)

        vel_dot_y = (vy - prev_vel[1]) / dt

        acc_sp = np.zeros(3)
        acc_sp[0] = MPC_XY_VEL_P_ACC * (vel_sp[0] - vx)
        acc_sp[1] = (
            MPC_XY_VEL_P_ACC * (vel_sp[1] - vy)
            + MPC_XY_VEL_I_ACC * vel_int[1]
            - MPC_XY_VEL_D_ACC * (
                prev_acc[1] + LPF * (vel_dot_y - prev_acc[1])
            )
        )
        acc_sp[2] = MPC_Z_VEL_P_ACC * (vel_sp[2] - vz)

        step = dt / 2.0

        for _ in range(2):
            s_old = s.copy()

            s[state_index["x"]] += s_old[state_index["vx"]] * step
            s[state_index["y"]] += s_old[state_index["vy"]] * step
            s[state_index["z"]] += s_old[state_index["vz"]] * step

            s[state_index["vx"]] += acc_sp[0] * step
            s[state_index["vy"]] += acc_sp[1] * step
            s[state_index["vz"]] += acc_sp[2] * step

        pred[h + 1] = s.copy()

        prev_acc = acc_sp.copy()

        vel_int[0] += (vel_sp[0] - s[state_index["vx"]]) * dt
        vel_int[1] += (vel_sp[1] - s[state_index["vy"]]) * dt
        vel_int[2] += (vel_sp[2] - s[state_index["vz"]]) * dt
        vel_int[2] = np.clip(vel_int[2], -A_OF_GRAVITY, A_OF_GRAVITY)

        prev_vel = np.array([
            s[state_index["vx"]],
            s[state_index["vy"]],
            s[state_index["vz"]],
        ])

    return pred


segments = []

if state in state_index:
    t_array = df["t"].to_numpy()

    start_idx = int(np.searchsorted(t_array, prediction_start_time))
    plot_every = max(1, int(round(prediction_interval / dt)))

    for i in range(start_idx, len(df), plot_every):
        row = df.iloc[i]
        pred = calc_prediction_one_row(i)

        # 予測線はその時刻 row["t"] から開始
        xs = row["t"] + np.arange(horizon + 1) * dt
        ys = pred[:, state_index[state]].copy()

        segments.append(np.column_stack([xs, ys]))


fig, ax = plt.subplots(figsize=(12, 5))
plt.subplots_adjust(bottom=0.25, right=0.88)

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
        colors=[(1.0, 0.0, 0.0, 0.35)],
        linewidths=1.5,
    )
    ax.add_collection(lc)

    ax.plot(
        [],
        [],
        color="red",
        linewidth=2,
        label=f"predicted_{state}",
    )

if target_col in df.columns:
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