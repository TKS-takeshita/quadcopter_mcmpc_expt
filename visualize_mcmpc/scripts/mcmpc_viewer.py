import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection

default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_20260501_083344.csv"

valid_states = [
    "e0","e1","e2","e3",
    "wx","wy","wz",
    "x","y","z",
    "vx","vy","vz",
    "input1","input2","input3","input4",
    "target_x","target_y","target_z",
    "cost"
]

state = sys.argv[1] if len(sys.argv) > 1 else "x"
csv_path = default_csv
mode = "all"

if len(sys.argv) > 2:
    if sys.argv[2] in ["all","1step"]:
        mode = sys.argv[2]
    else:
        csv_path = sys.argv[2]

if len(sys.argv) > 3:
    if sys.argv[3] in ["all","1step"]:
        mode = sys.argv[3]

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

t_cur = df["t"].values

fig, ax = plt.subplots(figsize=(12,5))
plt.subplots_adjust(bottom=0.25, right=0.88)

# =========================
# input
# =========================
if state.startswith("input"):
    col = "input_" + state[-1]
    ax.plot(t_cur, df[col], color="green", label=col)

# =========================
# target
# =========================
elif state.startswith("target"):
    ax.plot(t_cur, df[state], color="orange", linewidth=2, label=state)

# =========================
# cost
# =========================
elif state == "cost":
    ax.plot(t_cur, df["cost"], color="purple", linewidth=2, label="cost")

# =========================
# state
# =========================
else:
    cur_col = f"cur_{state}"
    ax.plot(t_cur, df[cur_col], color="blue", linewidth=2, label=cur_col)

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
            ax.plot(df["t_pred_1"], df[f"1{state}"],
                    color="red", linewidth=2, label="1step")

        else:
            segments = []
            colors = []

            for i in range(len(df)):
                xs = np.concatenate(([df.loc[i,"t"]],
                                     df.loc[i,pred_times_cols].values))
                ys = np.concatenate(([df.loc[i,cur_col]],
                                     df.loc[i,pred_state_cols].values))

                segments.append(np.column_stack([xs,ys]))
                alpha = 0.05 + 0.75*(i/max(1,len(df)-1))
                colors.append((1,0,0,alpha))

            lc = LineCollection(segments, colors=colors, linewidths=1.0)
            ax.add_collection(lc)

            ax.plot([],[],color="red",label="pred")

t_min = t_cur.min()
t_max = t_cur.max()

ax.set_title(state)
ax.set_xlabel("time")
ax.grid()
ax.legend()

ax.relim()
ax.autoscale_view()

# =========================
# 横スライダ
# =========================
init_width = min(2.0, t_max - t_min)
window_width = [init_width]

slider_ax = plt.axes([0.15,0.1,0.7,0.03])
time_slider = Slider(slider_ax,"time",t_min,t_max,
                     valinit=t_min + init_width/2)

# =========================
# 縦スライダ
# =========================
ymin, ymax = ax.get_ylim()
y_height = [ymax - ymin]

y_slider_ax = plt.axes([0.91,0.2,0.02,0.65])
y_slider = Slider(y_slider_ax,"y",ymin,ymax,
                  valinit=0.5*(ymin+ymax),
                  orientation="vertical")

def set_xlim(center):
    half = window_width[0]/2
    ax.set_xlim(center-half, center+half)
    fig.canvas.draw_idle()

def set_ylim(center):
    half = y_height[0]/2
    ax.set_ylim(center-half, center+half)
    fig.canvas.draw_idle()

def on_scroll(event):
    if event.inaxes != ax:
        return

    if event.key == "shift":
        scale = 0.8 if event.button=="up" else 1.25
        y_height[0] *= scale
        set_ylim(y_slider.val)
        return

    scale = 0.8 if event.button=="up" else 1.25
    window_width[0] *= scale
    set_xlim(time_slider.val)

time_slider.on_changed(lambda v:set_xlim(v))
y_slider.on_changed(lambda v:set_ylim(v))

fig.canvas.mpl_connect("scroll_event", on_scroll)

set_xlim(time_slider.val)
set_ylim(y_slider.val)

plt.show()