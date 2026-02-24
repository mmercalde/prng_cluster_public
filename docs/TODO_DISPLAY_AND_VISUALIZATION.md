# TODO: Display & Visualization Improvements

**Created:** December 6, 2025
**Priority:** HIGH - Next Major Task After Agent Manager
**Status:** PENDING

---

## Phase 1: Terminal Display (Real-time Monitoring)

### 1.1 Auto-attaching tmux - CRITICAL
**Problem:** Progress monitor launches in detached session, user must manually attach
**Current behavior:** `tmux attach -t prng_monitor` required
**Goal:** Auto-split terminal or auto-attach when running window_optimizer.py

**Files to modify:**
- `progress_monitor.py`
- `progress_display.py`
- `coordinator.py` (where tmux is spawned)

**Implementation ideas:**
- Check if already in tmux → split pane automatically
- If not in tmux → create new tmux session and attach
- Clean up stale sessions on startup

### 1.2 Live GPU Utilization
**Display:** Per-node progress bars showing activity
```
Zeus (2x 3080 Ti)  ████████████████████░░░░  85% [2 jobs]
rig-6600 (12x)     ██████████████░░░░░░░░░░  58% [7 jobs]
rig-6600b (12x)    █████████████░░░░░░░░░░░  54% [6 jobs]
```

**Data source:** coordinator.py job tracking, SSH heartbeats

### 1.3 Optuna Trial Progress
**Display:** Current trial params, best score, convergence trend
```
TRIAL 3/20 │ Best: 1638 │ Current: W313_O288_FT0.85_RT0.96
```

**Data source:** Optuna study object, trial callbacks

### 1.4 Survivor Counts (Live)
**Display:** Forward/Reverse/Bidirectional counts updating in real-time
```
SURVIVORS   Fwd: 1,245  │  Rev: 1,189  │  Bidir: 1,034
```

**Data source:** Sieve result files, accumulator dict

### 1.5 ETA Estimation
**Display:** Based on chunk completion rates
```
Forward ████████░░ 80%  ETA: 12s
Reverse ░░░░░░░░░░  0%  Waiting
```

**Data source:** Chunk timestamps, historical throughput

---

## Phase 2: Seaborn Visualizations (Post-run Analytics)

### 2.1 Threshold Optimization Heatmap - PRIORITY
**Purpose:** Visualize optimal threshold combinations
**Axes:**
- X-axis: forward_threshold (0.50 - 0.95)
- Y-axis: reverse_threshold (0.60 - 0.98)
- Color: survivor count or optimization score

**Data source:** Optuna study trials database

**Implementation:**
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Optuna trials
trials_df = study.trials_dataframe()

# Create pivot table
pivot = trials_df.pivot_table(
    values='value',  # score
    index='params_reverse_threshold',
    columns='params_forward_threshold',
    aggfunc='mean'
)

# Plot heatmap
sns.heatmap(pivot, annot=True, cmap='viridis')
plt.title('Threshold Optimization Surface')
plt.xlabel('Forward Threshold')
plt.ylabel('Reverse Threshold')
plt.savefig('threshold_heatmap.png')
```

### 2.2 Future Visualizations (Lower Priority)
- Optuna trial history (line plot of best score over trials)
- GPU performance comparison (bar chart: RTX 3080 Ti vs RX 6600)
- Survivor distribution (histogram of match rates)
- Parameter importance (Optuna's built-in importance plot)

---

## Technical Requirements

### Dependencies
```bash
pip install seaborn matplotlib pandas
# Already have: rich (for terminal display)
```

### File Structure
```
distributed_prng_analysis/
├── progress_display.py      # Terminal display (exists, needs fixes)
├── progress_monitor.py      # Monitor launcher (exists, needs tmux fix)
├── visualization/           # NEW directory
│   ├── __init__.py
│   ├── threshold_heatmap.py
│   └── post_run_report.py
```

---

## Acceptance Criteria

### Phase 1 Complete When:
- [ ] Running window_optimizer.py auto-opens progress monitor
- [ ] GPU utilization visible per-node
- [ ] Current trial and best score displayed
- [ ] Survivor counts update live
- [ ] ETA shown for sieve progress

### Phase 2 Complete When:
- [ ] Threshold heatmap generates after optimization run
- [ ] Heatmap saved to results/ directory
- [ ] Can identify optimal threshold region visually

---

## References

- Current progress_display.py: 616 lines, has WatcherProgress, ClusterProgress, PipelineProgress classes
- Current progress_monitor.py: Launches tmux session
- Rich library docs: https://rich.readthedocs.io/
- Seaborn heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html

