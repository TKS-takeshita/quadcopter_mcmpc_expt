import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection

default_csv = "/home/ros2/ws_mcmpc/src/quadcopter_mcmpc_position/csv/mcmpc_log_20260429_161843.csv"

state = sys.argv[1] if len(sys.argv) > 1 else "x"
csv_path = sys.argv[2] if len(sys.argv) > 2 else default_csv

valid_states = [
    "e0", "e1", "e2", "e3",
    "wx", "wy", "wz",
    "x", "y", "z",
    "vx", "vy", "vz",
    "input1", "input2", "input3", "input4"
]

if state not in valid_states:
    print(f"Invalid state: {state}")
    print("valid:", valid_states)
    sys.exit(1)

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

t_cur = df["t"].values

fig, ax = plt.subplots(figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)

# =========================
# input表示
# =========================
if state.startswith("input"):
    input_col = "input_" + state[-1]
    ax.plot(t_cur, df[input_col].values, color="green", label=input_col)

    t_min = t_cur.min()
    t_max = t_cur.max()

# =========================
# 状態表示
# =========================
else:
    cur_col = f"cur_{state}"

    # 現在状態：青
    ax.plot(t_cur, df[cur_col].values, color="blue", linewidth=2.0, label=cur_col)

    # horizon列を自動検出
    pred_times_cols = []
    pred_state_cols = []

    h = 1
    while True:
        t_col = f"t_pred_{h}"
        s_col = f"{h}{state}"   # 例: 1e0, 2e0, 1x, 2x

        if t_col not in df.columns or s_col not in df.columns:
            break

        pred_times_cols.append(t_col)
        pred_state_cols.append(s_col)
        h += 1

    if len(pred_times_cols) == 0:
        print("No horizon prediction columns found.")
        print("Expected columns like: t_pred_1, 1e0, t_pred_2, 2e0, ...")
        sys.exit(1)

    # 各制御周期の予測軌道を赤系で表示
    segments = []
    colors = []

    n_rows = len(df)

    for i in range(n_rows):
        xs = df.loc[i, pred_times_cols].values.astype(float)
        ys = df.loc[i, pred_state_cols].values.astype(float)

        points = np.column_stack([xs, ys])
        segments.append(points)

        # 古い制御周期ほど薄い赤
        alpha = 0.05 + 0.75 * (i / max(1, n_rows - 1))
        colors.append((1.0, 0.0, 0.0, alpha))

    lc = LineCollection(segments, colors=colors, linewidths=1.0)
    ax.add_collection(lc)

    ax.plot([], [], color="red", alpha=0.8, label="pred horizon")

    t_min = min(t_cur.min(), df[pred_times_cols].min().min())
    t_max = max(t_cur.max(), df[pred_times_cols].max().max())

ax.set_title(state)
ax.set_xlabel("time [s]")
ax.set_ylabel(state)
ax.grid(True)
ax.legend()

# autoscale
ax.relim()
ax.autoscale_view()

init_width = min(2.0, t_max - t_min)
window_width = [init_width]

slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
time_slider = Slider(
    slider_ax,
    "time",
    t_min,
    t_max,
    valinit=t_min + init_width / 2
)

def set_xlim(center):
    half = window_width[0] / 2
    left = max(t_min, center - half)
    right = min(t_max, center + half)

    if right - left < window_width[0]:
        if left <= t_min:
            right = min(t_max, t_min + window_width[0])
        if right >= t_max:
            left = max(t_min, t_max - window_width[0])

    ax.set_xlim(left, right)
    fig.canvas.draw_idle()

def zoom_ylim(scale):
    y_min, y_max = ax.get_ylim()
    center = 0.5 * (y_min + y_max)
    half = 0.5 * (y_max - y_min) * scale

    ax.set_ylim(center - half, center + half)
    fig.canvas.draw_idle()

def on_slider(val):
    set_xlim(val)

def on_scroll(event):
    if event.inaxes != ax:
        return

    # Shift + scroll：縦軸ズーム
    if event.key == "shift":
        if event.button == "up":
            zoom_ylim(0.8)
        elif event.button == "down":
            zoom_ylim(1.25)
        return

    # 通常scroll：横軸ズーム
    cur_xlim = ax.get_xlim()
    width = cur_xlim[1] - cur_xlim[0]

    if event.button == "up":
        width *= 0.8
    elif event.button == "down":
        width *= 1.25

    width = max(0.02, min(width, t_max - t_min))
    window_width[0] = width
    set_xlim(time_slider.val)

time_slider.on_changed(on_slider)
fig.canvas.mpl_connect("scroll_event", on_scroll)

set_xlim(time_slider.val)
plt.show()