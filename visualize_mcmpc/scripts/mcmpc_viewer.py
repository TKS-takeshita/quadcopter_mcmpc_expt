import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection

# default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_20260508_064854.csv"
default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_expt/quadcopter_mcmpc_position/csv/mcmpc_log_20260513_074656.csv"

valid_states = [
    "e0","e1","e2","e3",
    "wx","wy","wz",
    "x","y","z","yaw",
    "vx","vy","vz",
    "input1","input2","input3","input4",
    "target_x","target_y","target_z",
    "cost",

    "pos_sp_x","pos_sp_y","pos_sp_z",
    "vel_sp_x","vel_sp_y","vel_sp_z",
    "acc_sp_x","acc_sp_y","acc_sp_z",
    "yaw_sp","yawspeed_sp",

    "thrust_sp_x","thrust_sp_y","thrust_sp_z",

    "att_sp_w","att_sp_x","att_sp_y","att_sp_z",
    "rate_sp_x","rate_sp_y","rate_sp_z",
]

state = sys.argv[1] if len(sys.argv) > 1 else "x"
csv_path = default_csv
mode = "all"

if len(sys.argv) > 2:
    if sys.argv[2] in ["all", "1step"]:
        mode = sys.argv[2]
    else:
        csv_path = sys.argv[2]

if len(sys.argv) > 3:
    if sys.argv[3] in ["all", "1step"]:
        mode = sys.argv[3]

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

t_cur = df["t"].values

fig, ax = plt.subplots(figsize=(12, 5))
plt.subplots_adjust(bottom=0.25, right=0.88)

def has_col(col):
    return col in df.columns

def plot_col(col, color=None, linewidth=2, label=None, linestyle="-"):
    if has_col(col):
        ax.plot(
            t_cur,
            df[col],
            color=color,
            linewidth=linewidth,
            label=label or col,
            linestyle=linestyle,
        )
        return True

    print(f"[WARN] column not found: {col}")
    return False

# =========================
# calc vs topic
# topic を1サンプル前倒し
# =========================
def plot_calc_topic_pair(calc_col, topic_col):
    ok1 = False
    ok2 = False

    # calc
    if has_col(calc_col):
        ax.plot(
            t_cur,
            df[calc_col],
            color="blue",
            linewidth=2.0,
            label=calc_col,
        )
        ok1 = True

    # topic shifted
    if has_col(topic_col):
        shifted = df[topic_col].shift(-2)

        ax.plot(
            t_cur,
            shifted,
            color="green",
            linewidth=1.8,
            linestyle="--",
            label=topic_col + "_shift",
        )
        ok2 = True

    return ok1 or ok2

# =========================
# input
# =========================
if state.startswith("input"):
    col = "input_" + state[-1]
    plot_col(col, color="green", label=col)

# =========================
# target
# =========================
elif state.startswith("target"):
    plot_col(state, color="green", linewidth=2, label=state)

# =========================
# cost
# =========================
elif state == "cost":
    plot_col("cost", color="purple", linewidth=2, label="cost")

# =========================
# position setpoint
# =========================
elif state.startswith("pos_sp_"):
    axis = state[-1]

    plot_col(
        f"topic_pos_sp_{axis}",
        color="green",
        linewidth=2.0,
        label=f"topic_pos_sp_{axis}",
    )

# =========================
# velocity setpoint
# =========================
elif state.startswith("vel_sp_"):
    axis = state[-1]

    plot_calc_topic_pair(
        f"calc_vel_sp_{axis}",
        f"topic_vel_sp_{axis}",
    )

# =========================
# acceleration setpoint
# =========================
elif state.startswith("acc_sp_"):
    axis = state[-1]

    plot_calc_topic_pair(
        f"calc_acc_sp_{axis}",
        f"topic_acc_sp_{axis}",
    )

# =========================
# thrust setpoint
# =========================
elif state.startswith("thrust_sp_"):
    axis = state[-1]

    plot_calc_topic_pair(
        f"calc_thrust_sp_{axis}",
        f"topic_thrust_sp_{axis}",
    )

# =========================
# attitude setpoint
# =========================
elif state.startswith("att_sp_"):
    axis = state.split("_")[-1]   # w/x/y/z
    plot_calc_topic_pair(
        f"calc_att_sp_{axis}",
        f"topic_att_sp_{axis}",
    )

# =========================
# rate setpoint
# =========================
elif state.startswith("rate_sp_"):
    axis = state[-1]

    plot_calc_topic_pair(
        f"calc_rate_sp_{axis}",
        f"topic_rate_sp_{axis}",
    )

# =========================
# yaw setpoint
# =========================
elif state == "yaw_sp":

    plot_col(
        "topic_yaw_sp",
        color="green",
        linewidth=2.0,
        label="topic_yaw_sp",
    )

elif state == "yawspeed_sp":

    plot_col(
        "topic_yawspeed_sp",
        color="green",
        linewidth=2.0,
        label="topic_yawspeed_sp",
    )

# =========================
# yaw state
# =========================
elif state == "yaw":

    if all(has_col(c) for c in ["cur_e0", "cur_e1", "cur_e2", "cur_e3"]):

        q0 = df["cur_e0"]
        q1 = df["cur_e1"]
        q2 = df["cur_e2"]
        q3 = df["cur_e3"]

        yaw_cur = np.arctan2(
            2.0 * (q0*q3 + q1*q2),
            q0*q0 + q1*q1 - q2*q2 - q3*q3
        )

        ax.plot(
            t_cur,
            yaw_cur,
            color="blue",
            linewidth=3,
            label="cur_yaw",
        )

    plot_col(
        "topic_yaw_sp",
        color="green",
        linewidth=2.0,
        label="topic_yaw_sp",
        linestyle="--",
    )

    plot_col(
        "topic_att_sp_yaw",
        color="orange",
        linewidth=1.5,
        label="topic_att_sp_yaw",
        linestyle=":",
    )

# =========================
# normal state
# =========================
else:

    cur_col = f"cur_{state}"

    plot_col(
        cur_col,
        color="blue",
        linewidth=3,
        label=cur_col,
    )

    if state in ["x", "y", "z"]:

        plot_col(
            f"target_{state}",
            color="green",
            linewidth=2,
            label=f"target_{state}",
        )

        plot_col(
            f"topic_pos_sp_{state}",
            color="orange",
            linewidth=1.5,
            label=f"topic_pos_sp_{state}",
            linestyle="--",
        )

    pred_times_cols = []
    pred_state_cols = []

    h = 1

    while True:

        t_col = f"t_pred_{h}"
        s_col = f"{h}{state}"

        if t_col not in df.columns or s_col not in df.columns:
            break

        pred_times_cols.append(t_col)
        pred_state_cols.append(s_col)

        h += 1

    if len(pred_times_cols) > 0:

        if mode == "1step":

            ax.plot(
                df["t_pred_1"],
                df[f"1{state}"],
                color="red",
                linewidth=1.0,
                alpha=0.25,
                label="1step",
            )

        else:

            segments = []
            colors = []

            display_interval = 1.0
            last_display_time = -0.5

            for i in range(len(df)):

                current_time = df.loc[i, "t"]

                if current_time - last_display_time < display_interval:
                    continue

                last_display_time = current_time

                xs = np.concatenate((
                    [df.loc[i, "t"]],
                    df.loc[i, pred_times_cols].values
                ))

                ys = np.concatenate((
                    [df.loc[i, cur_col]],
                    df.loc[i, pred_state_cols].values
                ))

                segments.append(np.column_stack([xs, ys]))

                colors.append((1, 0, 0, 1.0))

            lc = LineCollection(
                segments,
                colors=colors,
                linewidths=2.5,
            )

            ax.add_collection(lc)

            ax.plot([], [], color="red", alpha=1.0, linewidth=2.5, label="pred")

# =========================
# axes
# =========================
t_min = t_cur.min()
t_max = t_cur.max()

ax.set_title(state)
ax.set_xlabel("time")

ax.grid()
ax.legend(fontsize=16)

ax.relim()
ax.autoscale_view()

# =========================
# horizontal slider
# =========================
init_width = min(2.0, t_max - t_min)

window_width = [init_width]

slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])

time_slider = Slider(
    slider_ax,
    "time",
    t_min,
    t_max,
    valinit=t_min + init_width / 2,
)

# =========================
# vertical slider
# =========================
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

    half = window_width[0] / 2

    ax.set_xlim(center - half, center + half)

    fig.canvas.draw_idle()

def set_ylim(center):

    half = y_height[0] / 2

    ax.set_ylim(center - half, center + half)

    fig.canvas.draw_idle()

def on_scroll(event):

    if event.inaxes != ax:
        return

    # y zoom
    if event.key == "shift":

        scale = 0.8 if event.button == "up" else 1.25

        y_height[0] *= scale

        set_ylim(y_slider.val)

        return

    # x zoom
    scale = 0.8 if event.button == "up" else 1.25

    window_width[0] *= scale

    set_xlim(time_slider.val)

# =========================
# keyboard control
# =========================
def on_key(event):

    # =====================
    # x move
    # =====================
    if event.key == "a":

        new_center = time_slider.val - window_width[0] * 0.1

        time_slider.set_val(new_center)

    elif event.key == "d":

        new_center = time_slider.val + window_width[0] * 0.1

        time_slider.set_val(new_center)

    # =====================
    # x zoom
    # =====================
    elif event.key == "w":

        window_width[0] *= 0.8

        set_xlim(time_slider.val)

    elif event.key == "p":

        window_width[0] *= 1.25

        set_xlim(time_slider.val)

    # =====================
    # y move
    # =====================
    elif event.key == "j":

        new_center = y_slider.val - y_height[0] * 0.1

        y_slider.set_val(new_center)

    elif event.key == "l":

        new_center = y_slider.val + y_height[0] * 0.1

        y_slider.set_val(new_center)

    # =====================
    # y zoom
    # =====================
    elif event.key == "i":

        y_height[0] *= 0.8

        set_ylim(y_slider.val)

    elif event.key == "k":

        y_height[0] *= 1.25

        set_ylim(y_slider.val)

    # =====================
    # reset
    # =====================
    elif event.key == "r":

        window_width[0] = init_width

        y_height[0] = ymax - ymin

        time_slider.set_val(t_min + init_width / 2)

        y_slider.set_val(0.5 * (ymin + ymax))

        set_xlim(time_slider.val)

        set_ylim(y_slider.val)

# connect
fig.canvas.mpl_connect("key_press_event", on_key)

time_slider.on_changed(lambda v: set_xlim(v))
y_slider.on_changed(lambda v: set_ylim(v))

fig.canvas.mpl_connect("scroll_event", on_scroll)

set_xlim(time_slider.val)
set_ylim(y_slider.val)

plt.show()