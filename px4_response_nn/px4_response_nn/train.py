import torch
import torch.nn as nn
import torch.optim as optim

from px4_response_nn.model import PX4ResponseNN
from px4_response_nn.dataset import load_dataset

def train(csv_path, dt, epochs=200):
    X, Y = load_dataset(csv_path, dt)

    model = PX4ResponseNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch {epoch}: {loss.item()}")

    torch.save(model.state_dict(), "model.pt")