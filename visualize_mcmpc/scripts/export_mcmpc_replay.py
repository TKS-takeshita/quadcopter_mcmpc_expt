#!/usr/bin/env python3
"""Export a CSV replay using the same Python port of the CUDA MCMPC model.

The CUDA sources are deliberately not modified; this utility calls the model
implementation already used by dynamics_mcmpc.py and writes a browser-friendly
JSON file for mcmpc_log_replay_viewer.html.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dynamics_mcmpc as model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output", default="mcmpc_replay.json")
    ap.add_argument("--horizon", type=int, default=model.PREDICTION_HORIZON)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()
    df = model.preprocess_log(pd.read_csv(args.csv), model.DT)
    indices = list(range(0, len(df), max(1, args.stride)))
    contexts = model.build_prediction_contexts(df, indices, model.DT)
    actual = []
    for i, row in df.iterrows():
        actual.append({"t": float(row.get("t", i*model.DT)),
                       "x": float(row["pos_x"]), "y": float(row["pos_y"]),
                       "z": float(row["pos_z"]), "vx": float(row["vel_x"]),
                       "vy": float(row["vel_y"]), "vz": float(row["vel_z"])})
    predictions = []
    for i in indices:
        ctx, prev = contexts.get(i, ({}, None))
        ctx = model.sync_context_from_logged_model(df.loc[i], ctx)
        logged = model.logged_motor_speed_from_row(df.loc[i])
        if logged is not None: prev = logged.copy()
        states, debug = model.rollout(df, i, args.horizon, model.DT, ctx, prev)
        rows = []
        for h, s in enumerate(states):
            rows.append({"t": float(df.loc[i, "t"] + h*model.DT),
                         "x": float(s[model.STATE_INDEX["x"]]),
                         "y": float(s[model.STATE_INDEX["y"]]),
                         "z": float(s[model.STATE_INDEX["z"]])})
        predictions.append({"start": int(i), "rows": rows})
    out = {"source": os.path.abspath(args.csv), "dt": model.DT,
           "actual": actual, "predictions": predictions}
    with open(args.output, "w", encoding="utf-8") as f: json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.output}: {len(actual)} samples, {len(predictions)} rollouts")
if __name__ == "__main__": main()
