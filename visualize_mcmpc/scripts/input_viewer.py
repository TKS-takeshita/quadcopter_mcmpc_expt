import argparse
import os
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider


default_csv = "/home/kt182/ws_mcmpc/src/quadcopter_mcmpc_expt/visualize_mcmpc/csv/mcmpc_log_move_square.csv"

dt = 0.02
prediction_start_time = 0.9
prediction_interval = 1.4

input_alias = {
    "ux": "x",
    "uy": "y",
    "uz": "z",
    "uyaw": "yaw",
}
input_names = ("x", "y", "z", "yaw")


def available_horizon(df, component):
    pattern = re.compile(rf"^u(\d+)_{re.escape(component)}$")
    indices = []
    for col in df.columns:
        match = pattern.match(col)
        if match is not None:
            indices.append(int(match.group(1)))
    if not indices:
        return 0
    return max(indices) + 1


def main():
    global dt, prediction_start_time, prediction_interval

    parser = argparse.ArgumentParser(description="View optimized MC-MPC input sequences.")
    parser.add_argument("input", nargs="?", default="ux", help="ux, uy, uz, uyaw, or x/y/z/yaw")
    parser.add_argument("csv_path", nargs="?", default=default_csv)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--dt", type=float, default=dt)
    parser.add_argument("--prediction-start", type=float, default=prediction_start_time)
    parser.add_argument("--prediction-interval", type=float, default=prediction_interval)
    args = parser.parse_args()

    component = input_alias.get(args.input, args.input)
    if component not in input_names:
        print("[ERROR] input must be one of: ux, uy, uz, uyaw, x, y, z, yaw")
        sys.exit(1)

    csv_path = args.csv_path
    dt = args.dt
    prediction_start_time = args.prediction_start
    prediction_interval = args.prediction_interval

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

    u0_col = f"u0_{component}"
    if u0_col not in df.columns:
        print(f"[ERROR] missing column: {u0_col}")
        sys.exit(1)

    max_horizon = available_horizon(df, component)
    horizon = max_horizon if args.horizon is None else min(args.horizon, max_horizon)
    if horizon <= 0:
        print(f"[ERROR] no uN_{component} columns found")
        sys.exit(1)

    print(f"[INFO] csv={csv_path}")
    print(
        f"[INFO] input=u_{component}, dt={dt:.6f}, horizon={horizon}, "
        f"prediction_start={prediction_start_time}, prediction_interval={prediction_interval}"
    )

    segments = []
    t_array = df["t"].to_numpy(float)
    start_idx = int(np.searchsorted(t_array, prediction_start_time))
    plot_every = max(1, int(round(prediction_interval / dt)))

    for row_idx in range(start_idx, len(df), plot_every):
        row = df.iloc[row_idx]
        xs = row["t"] + np.arange(horizon) * dt
        ys = np.full(horizon, np.nan)
        for h in range(horizon):
            col = f"u{h}_{component}"
            if col in df.columns:
                ys[h] = row[col]
        finite = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(finite) >= 2:
            segments.append(np.column_stack([xs[finite], ys[finite]]))

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25, right=0.88)

    ax.plot(
        df["t"],
        df[u0_col],
        color="gold",
        linewidth=2.5,
        label=f"current_optimal_{u0_col}",
    )

    if len(segments) > 0:
        lc = LineCollection(
            segments,
            colors="red",
            linewidths=1.5,
        )
        ax.add_collection(lc)
        ax.plot([], [], color="red", linewidth=2, label=f"predicted_u_{component}")

    ax.set_title(f"u_{component}")
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
    time_slider = Slider(slider_ax, "time", t_min, t_max, valinit=init_width / 2.0)

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


if __name__ == "__main__":
    main()
