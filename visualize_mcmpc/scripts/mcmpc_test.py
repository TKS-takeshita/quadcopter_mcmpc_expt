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
INTEGRATION_STEP = CONTROL_PERIOD / 2.0
N_OF_SAMPLES = 128 * 64
N_OF_ELITE = 10
ITERATION_TIMES = 2

A_OF_GRAVITY = 9.80665
MPC_VELD_LP = 5.0
LPF = (2.0 * np.pi * MPC_VELD_LP) / (1.0 + 2.0 * np.pi * MPC_VELD_LP)

SIGMA = np.array([0.01, 0.01, 0.01, 0.001], dtype=np.float64)

COST_Q = {
    "x": 10.0,
    "y": 10.0,
    "z": 10.0,
    "vx": 1.0,
    "vy": 1.0,
    "vz": 1.0,
    "e1": 0.0,
    "e2": 0.0,
    "e3": 0.0,
    "wx": 0.0,
    "wy": 0.0,
    "wz": 0.0,
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


def generate_samples(average_input: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, SIGMA, size=(n_samples, HORIZON, 4))
    return average_input[None, :, :] + noise


def simulate_batch(
    initial_state: np.ndarray,
    inputs: np.ndarray,
    target: dict,
    params: Params,
    return_predictions: bool,
) -> tuple[np.ndarray | None, np.ndarray]:
    n_samples = inputs.shape[0]
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

        vel_dot_y = (vy - prev_vy) / CONTROL_PERIOD
        acc_x = params.mpc_xy_vel_p_acc * (vel_sp_x - vx)
        acc_y = (
            params.mpc_xy_vel_p_acc * (vel_sp_y - vy)
            + params.mpc_xy_vel_i_acc * vel_int_y
            - params.mpc_xy_vel_d_acc * (prev_acc_y + LPF * (vel_dot_y - prev_acc_y))
        )
        acc_z = params.mpc_z_vel_p_acc * (vel_sp_z - vz)

        for _ in range(2):
            old_vx = vx.copy()
            old_vy = vy.copy()
            old_vz = vz.copy()
            x += old_vx * INTEGRATION_STEP
            y += old_vy * INTEGRATION_STEP
            z += old_vz * INTEGRATION_STEP
            vx += acc_x * INTEGRATION_STEP
            vy += acc_y * INTEGRATION_STEP
            vz += acc_z * INTEGRATION_STEP
            z_i += z * INTEGRATION_STEP

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
        prev_acc_x = acc_x.copy()
        prev_acc_y = acc_y.copy()
        prev_acc_z = acc_z.copy()
        vel_int_y += (vel_sp_y - vy) * CONTROL_PERIOD
        vel_int_z += (vel_sp_z - vz) * CONTROL_PERIOD
        vel_int_z = np.clip(vel_int_z, -A_OF_GRAVITY, A_OF_GRAVITY)

        if predictions is not None:
            predictions[:, h + 1, 0] = x
            predictions[:, h + 1, 1] = y
            predictions[:, h + 1, 2] = z

    _ = prev_vx, prev_vz, prev_acc_x, prev_acc_z
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
    capture_debug: bool = False,
) -> tuple[np.ndarray, float, DebugResult | None]:
    best_input = average_input.copy()
    debug = None
    min_cost = np.inf

    for iteration in range(ITERATION_TIMES):
        rng = np.random.default_rng(seed + iteration)
        samples = generate_samples(best_input, n_samples, rng)
        _, costs = simulate_batch(initial_state, samples, target, params, False)
        elite_indices = np.argsort(costs)[:N_OF_ELITE]
        elite_samples = samples[elite_indices]
        elite_costs = costs[elite_indices]
        best_input = weighted_average(elite_samples, elite_costs)
        min_cost = float(elite_costs[0])

        if capture_debug and iteration == ITERATION_TIMES - 1:
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
    x, y, z = state[IDX["x"]:IDX["z"] + 1]
    vx, vy, vz = state[IDX["vx"]:IDX["vz"] + 1]

    x_ref, y_ref, z_ref, _ = input_seq[0]
    vel_sp_x = params.mpc_xy_p * (x_ref - x)
    vel_sp_y = params.mpc_xy_p * (y_ref - y)
    vel_sp_z = params.mpc_z_p * (z_ref - z)

    acc_x = params.mpc_xy_vel_p_acc * (vel_sp_x - vx)
    acc_y = params.mpc_xy_vel_p_acc * (vel_sp_y - vy)
    acc_z = params.mpc_z_vel_p_acc * (vel_sp_z - vz)

    for _ in range(2):
        old_vx, old_vy, old_vz = vx, vy, vz
        x += old_vx * INTEGRATION_STEP
        y += old_vy * INTEGRATION_STEP
        z += old_vz * INTEGRATION_STEP
        vx += acc_x * INTEGRATION_STEP
        vy += acc_y * INTEGRATION_STEP
        vz += acc_z * INTEGRATION_STEP

    next_state[IDX["x"]:IDX["z"] + 1] = np.array([x, y, z])
    next_state[IDX["vx"]:IDX["vz"] + 1] = np.array([vx, vy, vz])
    return next_state


def run_rollout(
    initial_state: np.ndarray,
    target: dict,
    params: Params,
    duration: float,
    seed: int,
    n_samples: int,
) -> list[RolloutStep]:
    n_steps = max(1, int(np.ceil(duration / CONTROL_PERIOD))) + 1
    target_input = np.tile(
        np.array([target["x"], target["y"], target["z"], target["yaw"]], dtype=np.float64),
        (HORIZON, 1),
    )
    state = initial_state.copy()
    average_input = target_input.copy()
    steps = []

    for step in range(n_steps):
        shifted = shift_input(average_input)
        best_input, cost, _ = optimize(
            state,
            target,
            shifted,
            params,
            seed + step * 1000,
            n_samples,
            capture_debug=False,
        )
        steps.append(
            RolloutStep(
                t=step * CONTROL_PERIOD,
                state=state.copy(),
                average_input=shifted.copy(),
                best_input=best_input.copy(),
                cost=cost,
            )
        )
        state = apply_first_input(state, best_input, target, params)
        average_input = best_input

    return steps


def draw_debug(plots: dict, rollout: list[RolloutStep], target: dict, params: Params, step_idx: int, seed: int, n_samples: int):
    step = rollout[step_idx]
    _, _, debug = optimize(
        step.state,
        target,
        step.average_input,
        params,
        seed + step_idx * 1000,
        n_samples,
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

    for fig in plots["figures"].values():
        fig.canvas.draw_idle()


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
    parser.add_argument("--target", type=parse_vec4, default=parse_vec4("1.0,0.0,-0.8,0.0"), help="target x,y,z,yaw")
    parser.add_argument("--current", type=parse_vec3, default=parse_vec3("0.0,0.0,0.0"), help="initial/current x,y,z at t=0")
    parser.add_argument("--velocity", type=parse_vec3, default=parse_vec3("0.0,0.0,0.0"), help="initial vx,vy,vz")
    parser.add_argument("--yaw", type=float, default=0.0, help="initial yaw")
    parser.add_argument("--time", type=float, default=0.0, help="initial display time [s]")
    parser.add_argument("--duration", type=float, default=5.0, help="rollout duration from 0s [s]")
    parser.add_argument("--samples", type=int, default=N_OF_SAMPLES, help="number of input samples")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--simulation-params", action="store_true", help="use SIMULATION MPC_XY_P=0.30 instead of current default 0.95")
    args = parser.parse_args()
    if args.samples < N_OF_ELITE:
        parser.error(f"--samples must be >= {N_OF_ELITE}")

    params = Params(mpc_xy_p=0.30 if args.simulation_params else 0.95)
    target = make_target(args.target)
    initial_state = make_state(args.current, args.velocity, args.yaw)
    rollout = run_rollout(initial_state, target, params, args.duration, args.seed, args.samples)

    max_idx = len(rollout) - 1
    initial_idx = int(np.clip(round(args.time / CONTROL_PERIOD), 0, max_idx))
    plots = make_plots()
    draw_debug(plots, rollout, target, params, initial_idx, args.seed, args.samples)

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
        draw_debug(plots, rollout, target, params, step_idx, args.seed, args.samples)
        control_fig.canvas.draw_idle()

    time_slider.on_changed(on_time_change)
    plt.show()


if __name__ == "__main__":
    main()
