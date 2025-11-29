#!/usr/bin/env python3
"""
Train ONE kernel on survivor seeds — find the real PRNG
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load survivors
def load_seeds():
    fw = json.load(open('results/window_opt_forward_244_139.json'))
    rv = json.load(open('results/window_opt_reverse_244_139.json'))
    def extract(d): return {s['seed'] for r in d.get('results',[]) for s in r.get('survivors',[]) if 'seed' in s}
    return list(extract(fw) & extract(rv))

from prng_registry import KERNEL_REGISTRY, get_cpu_reference

WINDOW = 50
BATCH = 128

class SimpleDataset(Dataset):
    def __init__(self, seeds, kernel_name):
        self.seeds = seeds
        self.kernel = KERNEL_REGISTRY[kernel_name]['cpu_reference']
        print(f"Training on {len(seeds)} seeds with {kernel_name}")

    def __len__(self): return len(self.seeds)
    def __getitem__(self, idx):
        seed = self.seeds[idx]
        seq = self.kernel(seed, n=WINDOW+1, skip=0)
        seq = [x % 1000 for x in seq]
        return torch.tensor(seq[:-1]), seq[-1]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.head = nn.Linear(128, 1000)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        return self.head(h.squeeze(0))

def train_kernel(kernel_name, seeds):
    ds = SimpleDataset(seeds, kernel_name)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    losses = []
    for epoch in tqdm(range(20), desc=kernel_name):
        model.train()
        epoch_loss = 0
        for x, y in dl:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dl))
    return losses[-1], model

# Top 5 kernels to test
kernels_to_test = [
    'java_lcg_shift17',
    'xorshift32_hybrid',
    'pcg32',
    'mt19937',
    'xorshift64_hybrid'
]

seeds = load_seeds()
results = {}

for k in kernels_to_test:
    if k not in KERNEL_REGISTRY: continue
    final_loss, model = train_kernel(k, seeds)
    results[k] = final_loss
    torch.save(model.state_dict(), f"model_{k}.pt")

# Find best
best = min(results.items(), key=lambda x: x[1])
print(f"\nBEST KERNEL: {best[0]} (loss={best[1]:.4f})")

# Predict next 5
model = Net()
model.load_state_dict(torch.load(f"model_{best[0]}.pt"))
model.eval()

seed = random.choice(seeds)
seq = KERNEL_REGISTRY[best[0]]['cpu_reference'](seed, n=WINDOW, skip=0)
seq = [x % 1000 for x in seq]
cur = torch.tensor([seq])
draws = []
for _ in range(5):
    nxt = model(cur).argmax(1).item()
    draws.append(nxt)
    cur = torch.cat([cur[:, 1:], torch.tensor([[nxt]])], dim=1)

print(f"NEXT 5: {', '.join(f'{d:03d}' for d in draws)}")
