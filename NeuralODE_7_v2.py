"""
features: lag(5), rolling freq (20/50/100), gap, statistike prošlog kola
vremenski split: train + val (200) + back-test (100)
BEST + FINAL težine (validacioni loss)
back-test: hits/7, hit%, AUC, LRAP
predikcija iz stvarno poslednjeg kola
validacija 7 jedinstvenih, sortirano, 1..39 (bez random.randint fallback‑a)
snimanje u NeuralODE_7_v2_predikcija.txt
determinizam: PyTorch single-thread, use_deterministic_algorithms, seedovi za sve
Neural ODE arhitektura ostala (encoder → ODE → decoder, Euler integrator)
"""

"""
Neural Ordinary Differential Equations
PyTorch
"""

import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# =========================
# Seed (po pravilu projekta)
# =========================
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)


# =========================
# Konfiguracija
# =========================
CSV_PATH = "/loto7hh_4620_k41.csv"
OUT_TXT = Path("/NeuralODE_7_predikcija.txt")
N_MIN, N_MAX = 1, 39
K = 7
LAG = 5
WINDOWS = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 4621 #300
LR = 1e-3
ODE_STEPS = 50

T0 = time.time()
print()
print("START", datetime.today())
print()


# =========================
# 1) Učitaj CSV (sa headerom Num1..Num7)
# =========================
df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for i, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {i} nema 7 jedinstvenih brojeva: {row.tolist()}")
print(f"CSV: {CSV_PATH}  |  izvlačenja: {N}, brojeva po kolu: {K}")


# =========================
# 2) Multi-hot + feature engineering
# =========================
def draws_to_multihot(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def build_features(draws_arr: np.ndarray, y_multi: np.ndarray) -> np.ndarray:
    n, _ = draws_arr.shape
    lag_blocks = []
    for lag in range(1, LAG + 1):
        shifted = np.zeros_like(draws_arr)
        shifted[lag:] = draws_arr[:-lag]
        lag_blocks.append(shifted)
    lag_block = np.concatenate(lag_blocks, axis=1).astype(float)

    cum = np.cumsum(y_multi, axis=0)
    rolling_blocks = []
    for w in WINDOWS:
        rolled = np.zeros_like(cum, dtype=float)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        rolling_blocks.append(rolled / float(w))
    rolling_block = np.concatenate(rolling_blocks, axis=1)

    gap = np.zeros((n, N_MAX), dtype=float)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i in range(n):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in draws_arr[i]:
            last_seen[v - 1] = i

    prev = np.zeros_like(draws_arr)
    prev[1:] = draws_arr[:-1]
    s_sum = prev.sum(axis=1, keepdims=True).astype(float)
    s_odd = (prev % 2 == 1).sum(axis=1, keepdims=True).astype(float)
    s_low = (prev <= 19).sum(axis=1, keepdims=True).astype(float)
    s_rng = (prev.max(axis=1, keepdims=True) - prev.min(axis=1, keepdims=True)).astype(float)
    stats = np.concatenate([s_sum, s_odd, s_low, s_rng], axis=1)

    return np.concatenate([lag_block, rolling_block, gap, stats], axis=1)


Y_full = draws_to_multihot(draws)
X_full = build_features(draws, Y_full)
START = max(LAG, max(WINDOWS))

X_all = X_full[START:].astype(np.float32)
Y_all = Y_full[START:].astype(np.float32)

n_total = X_all.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_train_full, Y_train_full = X_all[:n_train], Y_all[:n_train]
X_tr, Y_tr = X_train_full[:-VAL_N], Y_train_full[:-VAL_N]
X_val, Y_val = X_train_full[-VAL_N:], Y_train_full[-VAL_N:]
X_back, Y_back = X_all[n_train:], Y_all[n_train:]
X_next_raw = X_full[-1:].astype(np.float32)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_val_s = scaler.transform(X_val).astype(np.float32)
X_back_s = scaler.transform(X_back).astype(np.float32)
X_next_s = scaler.transform(X_next_raw).astype(np.float32)

input_dim = X_tr_s.shape[1]
hidden_dim = 128

X_tr_t = torch.tensor(X_tr_s)
Y_tr_t = torch.tensor(Y_tr)
X_val_t = torch.tensor(X_val_s)
Y_val_t = torch.tensor(Y_val)
X_back_t = torch.tensor(X_back_s)
X_next_t = torch.tensor(X_next_s)


# =========================
# 3) Neural ODE dynamics i model (zadržava ideju iz starog koda)
# =========================
# Neural ODE dynamics
class ODEFunc(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# Euler integrator
def euler_integrate(func: nn.Module, h0: torch.Tensor, t0: float, t1: float, n_steps: int) -> torch.Tensor:
    t = t0
    h = h0
    dt = (t1 - t0) / n_steps
    for _ in range(n_steps):
        h = h + func(h) * dt
        t += dt
    return h


# Neural ODE model
class NeuralODERegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, out_dim)
        self.t0 = 0.0
        self.t1 = 1.0
        self.n_steps = ODE_STEPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.tanh(self.encoder(x))
        hT = euler_integrate(self.odefunc, h0, self.t0, self.t1, self.n_steps)
        return self.decoder(hT)


model = NeuralODERegressor(input_dim, N_MAX)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Weighted BCE za 7/39 neravnotežu
pos = float(Y_tr.sum() / Y_tr.size)
pos_weight = torch.tensor((1.0 - pos) / pos, dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# =========================
# 4) Treniranje + čuvanje BEST i FINAL težina
# =========================
print()
print("Treniranje modela ...")
"""
Treniranje modela ...
"""
print()

best_val = float("inf")
best_state = None
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_tr_t)
    loss = criterion(logits, Y_tr_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_loss = criterion(val_logits, Y_val_t).item()
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # print(f"Epoch {epoch}: Loss = {loss.item():.6f} val={val_loss:.6f}")

print()
if epoch == EPOCHS - 1:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}  best_val = {best_val:.6f}")
print()
"""
Epoch 4620: Loss = 0.384430  best_val = 1.149682
"""

final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
if best_state is None:
    best_state = final_state


# =========================
# 5) Pomoćne funkcije (top-K, metrike, opis)
# =========================
def topk_from_scores(scores_1d: np.ndarray, k: int = K) -> np.ndarray:
    scores = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -scores))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d: np.ndarray, y_true: np.ndarray) -> float:
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick: np.ndarray) -> str:
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


def predict_with(state: dict, X_t: torch.Tensor) -> np.ndarray:
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_t)).cpu().numpy()


# =========================
# 6) Back-test (BEST i FINAL)
# =========================
S_back_best = predict_with(best_state, X_back_t)
S_back_final = predict_with(final_state, X_back_t)

rows = [
    ("BEST", avg_hits(S_back_best, Y_back), safe_auc(Y_back, S_back_best), safe_lrap(Y_back, S_back_best)),
    ("FINAL", avg_hits(S_back_final, Y_back), safe_auc(Y_back, S_back_final), safe_lrap(Y_back, S_back_final)),
]
print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<8} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
for name, h, a, l in rows:
    print(f"{name:<8} {h:>8.3f} {100*h/K:>6.1f}% {a:>7.3f} {l:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


# =========================
# 7) Predikcija sledeće kombinacije
# =========================
pick_best = topk_from_scores(predict_with(best_state, X_next_t)[0])
pick_final = topk_from_scores(predict_with(final_state, X_next_t)[0])

for name, pick in [("BEST", pick_best), ("FINAL", pick_final)]:
    assert len(set(pick.tolist())) == K
    assert pick.min() >= N_MIN and pick.max() <= N_MAX
    assert list(pick) == sorted(pick.tolist())

print("🎯 Predikcija sledeće loto kombinacije:")
print(f"  BEST  -> {pick_best.tolist()}  ({describe(pick_best)})")
print(f"  FINAL -> {pick_final.tolist()}  ({describe(pick_final)})")
"""

"""


# =========================
# 8) Snimanje + ukupno vreme
# =========================
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N}, epochs={EPOCHS}) ---\n")
    f.write(f"BEST  -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"FINAL -> {pick_final.tolist()}  ({describe(pick_final)})\n")
print(f"Snimljeno u: {OUT_TXT}")

elapsed = time.time() - T0
print()
print("STOP", datetime.today())
print()
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()



"""
START 2026-05-24 16:46:44.531724

CSV: /loto7hh_4620_k41.csv  |  izvlačenja: 4620, brojeva po kolu: 7

Treniranje modela ...


Epoch 4620: Loss = 0.384430  best_val = 1.149682

Back-test (poslednjih 100 izvlačenja):
model      hits/7    hit%     AUC    LRAP
BEST        1.230   17.6%   0.504   0.248
FINAL       1.320   18.9%   0.523   0.266
(slučajan baseline ≈ 1.256 hits/7)

🎯 Predikcija sledeće loto kombinacije:
  BEST  -> [4, x, 16, y, 23, z, 32]  (suma=134, neparnih=2/7, niskih(<=19)=3/7, raspon=28)
  FINAL -> [19, x, 24, y, 29, z, 35]  (suma=188, neparnih=4/7, niskih(<=19)=1/7, raspon=16)
Snimljeno u: /NeuralODE_7_predikcija.txt

STOP 2026-05-24 16:57:17.693104

Ukupno vreme: 0:10:33  (633.2 s)
"""
