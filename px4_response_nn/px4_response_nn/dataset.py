import pandas as pd
import torch

def load_dataset(csv_path, dt):
    df = pd.read_csv(csv_path)

    X = []
    Y = []

    for i in range(len(df) - 1):
        cur = df.iloc[i]
        nxt = df.iloc[i + 1]

        x = [
            cur.e0, cur.e1, cur.e2, cur.e3,
            cur.wx, cur.wy, cur.wz,
            cur.input1, cur.input2, cur.input3, cur.input4
        ]

        az = (nxt.vz - cur.vz) / dt

        y = [
            nxt.wx, nxt.wy, nxt.wz,
            az
        ]

        X.append(x)
        Y.append(y)

    return torch.tensor(X, dtype=torch.float32), \
           torch.tensor(Y, dtype=torch.float32)