import math, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- data: predict next sine value from previous window ---
def make_sine_dataset(n_samples=2000, seq_len=32):
    xs, ys = [], []
    for k in range(n_samples):
        phase = torch.rand(1).item() * 2*math.pi
        t = torch.linspace(0, 4*math.pi, steps=seq_len+1) + phase
        s = torch.sin(t) + 0.05*torch.randn_like(t)
        xs.append(s[:-1].unsqueeze(-1))  # [seq_len, 1]
        ys.append(s[1:].unsqueeze(-1))   # next-step target
    X = torch.stack(xs)  # [N, T, 1]
    Y = torch.stack(ys)  # [N, T, 1]
    return X, Y

X, Y = make_sine_dataset(n_samples=4000, seq_len=32)
train_ds = TensorDataset(X[:3600], Y[:3600])
val_ds   = TensorDataset(X[3600:], Y[3600:])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128)

# --- model ---
class TinyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        # (nice-to-have) init forget bias to help early training
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                # forget gate is second quarter in [i, f, g, o]
                hidden_size = bias.shape[0] // 4
                with torch.no_grad():
                    bias[hidden_size:2*hidden_size].fill_(1.0)

    def forward(self, x):
        # x: [B, T, 1]
        h, _ = self.lstm(x)         # [B, T, H]
        y = self.head(h)            # [B, T, 1]
        return y

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyLSTM().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.MSELoss()

def evaluate(loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total += lossf(pred, yb).item() * xb.size(0)
    return total / len(loader.dataset)

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = lossf(pred, yb)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    val_loss = evaluate(val_loader)
    print(f"epoch {epoch+1:02d}  val_mse={val_loss:.5f}")
