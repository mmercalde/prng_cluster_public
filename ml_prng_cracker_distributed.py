#!/usr/bin/env python3
"""
ML PRNG Cracker – FINAL, NO POST-TRAIN CRASH
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

os.environ['MASTER_PORT'] = '29504'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def load_seeds():
    if dist.get_rank() == 0:
        fw = json.load(open('results/window_opt_forward_244_139.json'))
        rv = json.load(open('results/window_opt_reverse_244_139.json'))
        def extract(d): return {s['seed'] for r in d.get('results',[]) for s in r.get('survivors',[]) if 'seed' in s}
        seeds = list(extract(fw) & extract(rv))
        print(f"[Rank 0] Loaded {len(seeds):,} seeds")
    else:
        seeds = None
    obj = [seeds]
    dist.broadcast_object_list(obj, 0)
    return obj[0]

from prng_registry import KERNEL_REGISTRY, get_cpu_reference

WINDOW = 50
BATCH  = 128

class LiveDataset(Dataset):
    def __init__(self, seeds):
        self.seeds = seeds
        self.kernels = [(n, c) for n, c in KERNEL_REGISTRY.items() if c.get('cpu_reference')]
        print(f"[Rank {dist.get_rank()}] {len(seeds)} seeds × {len(self.kernels)} kernels")

    def __len__(self): return len(self.seeds) * len(self.kernels)
    def __getitem__(self, idx):
        s_idx = idx // len(self.kernels)
        k_idx = idx % len(self.kernels)
        seed = self.seeds[s_idx]
        name, cfg = self.kernels[k_idx]
        try:
            seq = cfg['cpu_reference'](seed, n=WINDOW+1, skip=0)
            seq = [x % 1000 for x in seq]
            return torch.tensor(seq[:-1]), seq[-1], name
        except:
            seq = [random.randint(0,999) for _ in range(WINDOW+1)]
            return torch.tensor(seq[:-1]), seq[-1], name

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 128)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, batch_first=True), 3)
        self.head = nn.Linear(128, 1000)
        pe = torch.zeros(WINDOW, 128)
        pos = torch.arange(WINDOW).unsqueeze(1)
        div = torch.exp(torch.arange(0,128,2) * -(8.0/WINDOW))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.emb(x) + self.pe
        x = self.tf(x)
        return self.head(x[:, -1])

def train(rank, world_size):
    setup(rank, world_size)
    seeds = load_seeds()
    local_seeds = seeds[rank::world_size]

    ds = LiveDataset(local_seeds)
    sampler = DistributedSampler(ds, world_size, rank)
    dl = DataLoader(ds, batch_size=BATCH, sampler=sampler, num_workers=0)

    model = Net().to(rank)
    model = DDP(model, device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()

    pbar = tqdm(range(15), desc=f"Rank {rank}", position=rank)
    for epoch in pbar:
        model.train()
        sampler.set_epoch(epoch)
        loss_sum = count = 0
        for x, y, _ in dl:
            x, y = x.to(rank), y.to(rank)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            count += 1
        pbar.set_postfix({"loss": f"{loss_sum/count:.4f}"})

    if rank == 0:
        torch.save(model.module.state_dict(), "model.pt")
        print("\nModel saved")
    cleanup()

# --------------------------------------------------------------
# VALIDATION & PREDICTION — RUN ONLY ON RANK 0
# --------------------------------------------------------------
def validate_and_predict():
    # Re-init dist for rank 0 only
    world_size = 1
    setup(0, world_size)

    model = Net()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    seeds = load_seeds()
    acc = {}
    print("Validating...")
    for name, cfg in tqdm(KERNEL_REGISTRY.items(), desc="Kernels"):
        if not cfg.get('cpu_reference'): continue
        correct = total = 0
        for seed in random.sample(seeds, 2000 // len([k for k in KERNEL_REGISTRY if KERNEL_REGISTRY[k].get('cpu_reference')]) + 1):
            try:
                seq = cfg['cpu_reference'](seed, n=WINDOW+1, skip=0)
                seq = [x % 1000 for x in seq]
                x = torch.tensor([seq[:-1]])
                y = seq[-1]
                pred = model(x).argmax(1).item()
                correct += pred == y
                total += 1
            except: continue
        if total: acc[name] = correct / total

    print("\nACCURACY")
    for k in sorted(acc, key=acc.get, reverse=True):
        print(f"{k:30} {acc[k]*100:6.2f}%")

    best = max(acc.items(), key=lambda x: x[1])
    print(f"\nBEST → {best[0]} ({best[1]*100:.2f}%)")

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

    cleanup()

def main():
    world_size = torch.cuda.device_count() or 1
    print(f"GPUs: {world_size}")

    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 1)

    # Run validation only on rank 0
    if world_size == 1 or os.getenv('RANK') == '0':
        validate_and_predict()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
