# Phase 7 Soak Test Plan
## WATCHER Dispatch — Sustained Validation
**Version:** 1.0.0
**Date:** 2026-02-03
**Status:** Ready for Execution
**Prerequisite:** Phase 7 D5 End-to-End (PASSED Session 59)

---

## 1. Purpose

D5 proved the wiring works for a single cycle. Soak testing proves the wiring **survives sustained operation** — the difference between "it ran once" and "it can run autonomously for days."

Three tests target three distinct failure classes:

| Test | Failure Class | Duration |
|------|--------------|----------|
| **A: Daemon Endurance** | Resource leaks (memory, file handles, VRAM) | 2-4 hours |
| **B: Sequential Requests** | Queue corruption, race conditions, archive integrity | 5-10 requests (~30-60 min) |
| **C: Sustained Autonomous Loop** | Full-cycle degradation, learning feedback correctness | 2+ hours (continuous cycles) |

**Exit criterion:** All three pass → autonomous loop declared production-ready.

---

## 2. Prerequisites (Run Once Before Any Soak Test)

### 2.1 Clean Starting State

```bash
cd ~/distributed_prng_analysis

# Verify no halt flag
ls -la watcher_halt.flag 2>/dev/null && echo "HALT FLAG EXISTS — remove first" || echo "OK: no halt flag"

# Verify no stale lock files
find . -name "*.lock" -mmin +60 -ls

# Clear old watcher request archives (keep queue clean)
ls watcher_requests/archive/ 2>/dev/null | wc -l

# Check disk space (logs + archives grow during soak)
df -h /home/michael
```

### 2.2 Enable Test Mode + Bootstrap Synthetic History

Only needed if `lottery_history.json` doesn't exist or you want a fresh start:

```bash
cd ~/distributed_prng_analysis

# Enable test mode
python3 -c "
import json
with open('watcher_policies.json') as f: p = json.load(f)
p['test_mode'] = True
p['synthetic_injection']['enabled'] = True
with open('watcher_policies.json','w') as f: json.dump(p,f,indent=2)
print('Test mode enabled. True seed:', p['synthetic_injection']['true_seed'])
"

# Check if lottery_history.json exists
if [ ! -f lottery_history.json ]; then
    echo "Bootstrapping synthetic history..."
    python3 << 'BOOTSTRAP'
import json
from datetime import datetime, timedelta
from prng_registry import get_cpu_reference

TRUE_SEED = 12345
PRNG_TYPE = "java_lcg"
NUM_DRAWS = 5000
MOD = 1000

prng = get_cpu_reference(PRNG_TYPE)
raw = prng(TRUE_SEED, NUM_DRAWS)
vals = [v % MOD for v in raw]

draws = []
date = datetime(2024,1,1)

for i,v in enumerate(vals):
    digits = [(v//100)%10,(v//10)%10,v%10]
    session = "midday" if i%2==0 else "evening"
    draws.append({
        "date": date.strftime("%Y-%m-%d"),
        "session": session,
        "draw": digits,
        "value": v,
        "draw_source": "synthetic_bootstrap",
        "true_seed": TRUE_SEED
    })
    if session=="evening":
        date += timedelta(days=1)

with open("lottery_history.json","w") as f:
    json.dump({"draws":draws},f,indent=2)

with open("optimal_window_config.json","w") as f:
    json.dump({
        "prng_type": PRNG_TYPE,
        "mod": MOD,
        "bootstrap": True
    },f,indent=2)

print("Bootstrap complete:", len(draws), "draws")
BOOTSTRAP
else
    echo "lottery_history.json exists — skipping bootstrap"
fi
```

### 2.3 Verify LLM Server

```bash
# Check if DeepSeek-R1-14B is running
curl -s http://localhost:8080/health | head -5

# If not running, start it
cd ~/distributed_prng_analysis
bash llm_services/start_llm_servers.sh

# Verify grammar loading works
curl -s -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","max_tokens":5}' | head -3
```

### 2.4 Verify Pipeline Artifacts Exist

The daemon and dispatch functions expect certain output files from previous pipeline runs:

```bash
cd ~/distributed_prng_analysis

echo "=== Checking Required Artifacts ==="
for f in optimal_window_config.json \
         bidirectional_survivors.json \
         optimal_scorer_config.json \
         survivors_with_scores.json; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(stat --format='%s' $f) bytes)"
    else
        echo "  MISSING: $f — run pipeline Steps 1-3 first"
    fi
done
```

If artifacts are missing, run a quick pipeline first:
```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 3
```

---

## 3. Monitoring Script (Shared Across All Tests)

Save this as `soak_monitor.sh` on Zeus. Run it in a **separate terminal** during all soak tests.

```bash
#!/bin/bash
# soak_monitor.sh — Resource monitoring for soak tests
# Usage: bash soak_monitor.sh [interval_seconds]

INTERVAL=${1:-30}
LOGFILE="logs/soak_monitor_$(date +%Y%m%d_%H%M%S).log"
PROJECT=~/distributed_prng_analysis

mkdir -p "$PROJECT/logs"

echo "=== SOAK MONITOR STARTED $(date) ===" | tee "$LOGFILE"
echo "Interval: ${INTERVAL}s | Log: $LOGFILE"
echo ""

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "--- $TS ---" >> "$LOGFILE"
    
    # 1. Python memory (RSS of watcher_agent processes)
    WATCHER_PID=$(pgrep -f "watcher_agent" | head -1)
    if [ -n "$WATCHER_PID" ]; then
        RSS_KB=$(ps -o rss= -p "$WATCHER_PID" 2>/dev/null)
        RSS_MB=$((RSS_KB / 1024))
        echo "  WATCHER RSS: ${RSS_MB} MB (PID $WATCHER_PID)" | tee -a "$LOGFILE"
    else
        echo "  WATCHER: not running" | tee -a "$LOGFILE"
    fi
    
    # 2. GPU VRAM (NVIDIA)
    VRAM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$VRAM" ]; then
        echo "  GPU VRAM: $VRAM" | tee -a "$LOGFILE"
    fi
    
    # 3. Open file handles
    if [ -n "$WATCHER_PID" ]; then
        FD_COUNT=$(ls /proc/$WATCHER_PID/fd 2>/dev/null | wc -l)
        echo "  File descriptors: $FD_COUNT" | tee -a "$LOGFILE"
    fi
    
    # 4. Disk space
    DISK_AVAIL=$(df -h /home/michael --output=avail | tail -1 | tr -d ' ')
    echo "  Disk available: $DISK_AVAIL" | tee -a "$LOGFILE"
    
    # 5. Watcher request queue
    PENDING=$(ls "$PROJECT/watcher_requests/"*.json 2>/dev/null | grep -v archive | wc -l)
    ARCHIVED=$(ls "$PROJECT/watcher_requests/archive/"*.json 2>/dev/null | wc -l)
    echo "  Requests pending: $PENDING | archived: $ARCHIVED" | tee -a "$LOGFILE"
    
    # 6. Decision log growth
    DECISIONS=$(wc -l < "$PROJECT/watcher_decisions.jsonl" 2>/dev/null || echo 0)
    echo "  Decisions logged: $DECISIONS" | tee -a "$LOGFILE"
    
    # 7. LLM server status
    LLM_UP=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null)
    echo "  LLM health: HTTP $LLM_UP" | tee -a "$LOGFILE"
    
    # 8. Any errors in watcher log (last N seconds)
    RECENT_ERRORS=$(tail -50 "$PROJECT/logs/watcher_agent.log" 2>/dev/null | grep -ci "error\|exception\|traceback")
    echo "  Recent errors in log: $RECENT_ERRORS" | tee -a "$LOGFILE"
    
    echo "" | tee -a "$LOGFILE"
    sleep "$INTERVAL"
done
```

**Make executable:**
```bash
chmod +x ~/distributed_prng_analysis/soak_monitor.sh
```

---

## 4. Soak Test A: Daemon Endurance (2-4 Hours)

### Goal
Verify the WATCHER daemon can idle and respond without resource degradation over hours.

### What It Tests
- Python process memory stability (no slow leaks)
- File handle lifecycle (open/close properly)
- LLM connection resilience (reconnects after timeouts)
- Log rotation sanity (doesn't fill disk)
- Halt flag responsiveness (can stop cleanly at any time)

### Procedure

**Terminal 1 — Monitor:**
```bash
cd ~/distributed_prng_analysis
bash soak_monitor.sh 60
```

**Terminal 2 — Daemon:**
```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --daemon 2>&1 | tee logs/soak_A_$(date +%Y%m%d_%H%M%S).log
```

**Terminal 3 — Baseline + Periodic Checks:**
```bash
cd ~/distributed_prng_analysis

# Record baseline memory
echo "=== BASELINE $(date) ==="
WATCHER_PID=$(pgrep -f "watcher_agent" | head -1)
ps -o pid,rss,vsz,%mem,etime -p "$WATCHER_PID"

# Check every 30 min:
# 1. Memory growth
ps -o pid,rss,vsz,%mem,etime -p "$WATCHER_PID"

# 2. File descriptors
ls /proc/$WATCHER_PID/fd | wc -l

# 3. Decision log entries
wc -l watcher_decisions.jsonl

# 4. Any Python tracebacks
grep -c "Traceback" logs/watcher_agent.log
```

**After 2-4 hours — Halt cleanly:**
```bash
python3 -m agents.watcher_agent --halt "Soak Test A complete"

# Then clear halt for next test
rm watcher_halt.flag
```

### Pass Criteria

| Metric | Pass | Fail |
|--------|------|------|
| Memory growth | < 50 MB over 4 hours | > 200 MB growth (leak) |
| File descriptors | Stable ±5 | Monotonically increasing |
| Tracebacks in log | 0 | Any unhandled exception |
| LLM reconnection | Handles timeout gracefully | Connection permanently lost |
| Halt responsiveness | Stops within 30s | Hangs or doesn't stop |
| Disk usage growth | < 50 MB logs | > 500 MB (log flooding) |

### Known Risks
- If no requests arrive, daemon is mostly idle — memory leaks may be slow. Inject 1-2 manual requests at the 1h and 2h marks to exercise the dispatch path:

```bash
# Manual request injection during test
cat > watcher_requests/soak_probe_$(date +%s).json << 'EOF'
{
    "request_type": "selfplay",
    "source": "soak_test_A_probe",
    "timestamp": "2026-02-03T12:00:00Z",
    "parameters": {
        "episodes": 2,
        "policy_conditioned": true
    },
    "reason": "Soak test A — daemon responsiveness probe"
}
EOF
echo "Probe request injected at $(date)"
```

---

## 5. Soak Test B: Sequential Request Handling (5-10 Requests)

### Goal
Verify the request queue processes multiple sequential requests without corruption, race conditions, or lost requests.

### What It Tests
- Request pickup ordering (FIFO)
- Archive integrity (every processed request moves to `archive/`)
- No duplicate processing (each request handled exactly once)
- Correct dispatch type routing (selfplay vs learning loop)
- LLM lifecycle transitions across consecutive dispatches

### Procedure

**Step 1 — Prepare 10 test requests:**

```bash
cd ~/distributed_prng_analysis

# Clear any existing test requests
mkdir -p watcher_requests/archive

# Generate 10 sequential requests with mix of types
python3 << 'GENREQS'
import json, os
from datetime import datetime, timedelta

base_time = datetime(2026, 2, 3, 10, 0, 0)

requests = [
    # 5 selfplay requests (varying episode counts)
    {"type": "selfplay", "episodes": 2, "reason": "Soak B — request 01 (selfplay 2ep)"},
    {"type": "selfplay", "episodes": 3, "reason": "Soak B — request 02 (selfplay 3ep)"},
    {"type": "selfplay", "episodes": 2, "reason": "Soak B — request 03 (selfplay 2ep)"},
    {"type": "selfplay", "episodes": 1, "reason": "Soak B — request 04 (selfplay 1ep)"},
    {"type": "selfplay", "episodes": 2, "reason": "Soak B — request 05 (selfplay 2ep)"},
    # 3 learning loop requests
    {"type": "learning_loop", "scope": "steps_3_5_6", "reason": "Soak B — request 06 (learn 3-5-6)"},
    {"type": "learning_loop", "scope": "steps_3_5_6", "reason": "Soak B — request 07 (learn 3-5-6)"},
    {"type": "learning_loop", "scope": "steps_3_5_6", "reason": "Soak B — request 08 (learn 3-5-6)"},
    # 2 more selfplay (interleaved)
    {"type": "selfplay", "episodes": 2, "reason": "Soak B — request 09 (selfplay 2ep)"},
    {"type": "selfplay", "episodes": 3, "reason": "Soak B — request 10 (selfplay 3ep)"},
]

for i, req in enumerate(requests):
    ts = base_time + timedelta(minutes=i)
    filename = f"soak_B_{i+1:02d}_{ts.strftime('%Y%m%d_%H%M%S')}.json"
    
    payload = {
        "request_type": req["type"],
        "source": "soak_test_B",
        "sequence_number": i + 1,
        "timestamp": ts.isoformat() + "Z",
        "reason": req["reason"],
        "parameters": {}
    }
    
    if req["type"] == "selfplay":
        payload["parameters"]["episodes"] = req["episodes"]
        payload["parameters"]["policy_conditioned"] = True
    elif req["type"] == "learning_loop":
        payload["parameters"]["scope"] = req["scope"]
    
    with open(f"watcher_requests/{filename}", "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"  Created: {filename}")

print(f"\nTotal: {len(requests)} requests staged in watcher_requests/")
GENREQS
```

**Step 2 — Start daemon and monitor:**

Terminal 1:
```bash
cd ~/distributed_prng_analysis
bash soak_monitor.sh 15
```

Terminal 2:
```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --daemon 2>&1 | tee logs/soak_B_$(date +%Y%m%d_%H%M%S).log
```

**Step 3 — Watch processing (Terminal 3):**

```bash
cd ~/distributed_prng_analysis

# Live-watch the request queue drain
watch -n 5 'echo "=== PENDING ===" && ls watcher_requests/*.json 2>/dev/null | grep -v archive | head -20 && echo "" && echo "=== ARCHIVED ===" && ls watcher_requests/archive/ 2>/dev/null | wc -l'
```

**Step 4 — Post-test validation:**

```bash
cd ~/distributed_prng_analysis

echo "=== SOAK TEST B VALIDATION ==="
echo ""

# 1. All 10 requests archived?
ARCHIVED=$(ls watcher_requests/archive/soak_B_*.json 2>/dev/null | wc -l)
echo "Archived: $ARCHIVED / 10"

# 2. Any requests still pending?
PENDING=$(ls watcher_requests/soak_B_*.json 2>/dev/null | wc -l)
echo "Still pending: $PENDING (should be 0)"

# 3. Check decision log for all 10
echo ""
echo "Decision log entries for soak_B:"
grep -c "soak_test_B" watcher_decisions.jsonl

# 4. Check for errors
echo ""
echo "Errors during test:"
grep -i "error\|exception\|traceback" logs/soak_B_*.log | head -20

# 5. Verify no duplicate processing
echo ""
echo "Unique sequence numbers processed:"
grep "soak_test_B" watcher_decisions.jsonl | python3 -c "
import sys, json
seen = set()
dupes = 0
for line in sys.stdin:
    try:
        d = json.loads(line)
        seq = d.get('sequence_number', 'unknown')
        if seq in seen:
            dupes += 1
            print(f'  DUPLICATE: sequence {seq}')
        seen.add(seq)
    except: pass
print(f'Unique: {len(seen)} | Duplicates: {dupes}')
"
```

### Pass Criteria

| Metric | Pass | Fail |
|--------|------|------|
| Requests archived | 10 / 10 | Any missing |
| Requests pending after test | 0 | Any stuck |
| Duplicate processing | 0 | Any duplicates |
| Correct routing | Selfplay → selfplay, Learn → learn | Misrouted |
| LLM lifecycle | Clean stop/restart between dispatches | Hung LLM or VRAM leak |
| Decision log | 10 entries with unique sequence numbers | Missing or corrupt entries |
| Errors | 0 unhandled exceptions | Any traceback |

### Alternative: No-Daemon Batch Mode

If the daemon's queue-scanning interval is slow, you can also test sequentially via `--process-requests`:

```bash
# Inject all 10, then process them one batch at a time
PYTHONPATH=. python3 agents/watcher_agent.py --process-requests
# Wait for completion, then re-run until queue is empty
```

---

## 6. Soak Test C: Sustained Autonomous Loop (2+ Hours)

### Goal
Validate the **complete autonomous cycle** under continuous load: new draw → diagnostics → trigger evaluation → dispatch (selfplay or learning loop) → evaluation → archive → repeat.

This is the definitive test. If this passes, the autonomous loop is production-ready.

### What It Tests
- Synthetic draw injection continuity
- Chapter 13 diagnostics generation under load
- Trigger evaluation accuracy (does the right thing fire?)
- Full dispatch cycle (selfplay + learning loop interleaved)
- LLM lifecycle under sustained stop/start cycling
- Policy candidate accumulation
- Telemetry integrity over time
- Convergence signal (hit rate should trend up with known seed)

### Procedure

**Step 1 — Configure continuous injection:**

```bash
cd ~/distributed_prng_analysis

# Set synthetic injection to fire every 2 minutes
python3 -c "
import json
with open('watcher_policies.json') as f: p = json.load(f)
p['test_mode'] = True
p['synthetic_injection']['enabled'] = True
p['synthetic_injection']['interval_seconds'] = 120  # 2 min between draws
with open('watcher_policies.json','w') as f: json.dump(p,f,indent=2)
print('Injection interval: 120s')
print('True seed:', p['synthetic_injection']['true_seed'])
"
```

**Step 2 — Snapshot pre-test state:**

```bash
cd ~/distributed_prng_analysis

echo "=== PRE-SOAK-C SNAPSHOT ==="
echo "Draws in history: $(python3 -c \"import json; d=json.load(open('lottery_history.json')); print(len(d['draws']))\")"
echo "Policy candidates: $(ls policy_history/ 2>/dev/null | wc -l)"
echo "Decision log lines: $(wc -l < watcher_decisions.jsonl 2>/dev/null || echo 0)"
echo "Archived requests: $(ls watcher_requests/archive/ 2>/dev/null | wc -l)"
echo "Selfplay candidates: $(ls learned_policy_candidate.json 2>/dev/null && echo 'exists' || echo 'none')"
echo "Telemetry: $(cat telemetry/learning_health_latest.json 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f\"{d.get(\"total_models\",0)} models\")' 2>/dev/null || echo 'no telemetry')"
date > logs/soak_C_start_timestamp.txt
```

**Step 3 — Launch daemon + synthetic injector:**

Terminal 1 (Monitor):
```bash
cd ~/distributed_prng_analysis
bash soak_monitor.sh 30
```

Terminal 2 (Daemon):
```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --daemon 2>&1 | tee logs/soak_C_$(date +%Y%m%d_%H%M%S).log
```

Terminal 3 (Synthetic Draw Injector — if not integrated into daemon):
```bash
cd ~/distributed_prng_analysis
# If the daemon auto-injects (check watcher_policies.json synthetic_injection),
# this terminal just watches. Otherwise run:
python3 synthetic_draw_injector.py --daemon --interval 120 2>&1 | tee logs/soak_C_injector.log
```

Terminal 4 (Live cycle counter):
```bash
cd ~/distributed_prng_analysis

# Count completed cycles every 60s
while true; do
    CYCLES=$(grep -c "COMPLETED" watcher_decisions.jsonl 2>/dev/null || echo 0)
    DRAWS=$(python3 -c "import json; d=json.load(open('lottery_history.json')); print(len(d['draws']))" 2>/dev/null)
    CANDIDATES=$(ls policy_history/ 2>/dev/null | wc -l)
    
    echo "[$(date '+%H:%M:%S')] Cycles: $CYCLES | Draws: $DRAWS | Policy candidates: $CANDIDATES"
    sleep 60
done
```

**Step 4 — Let it run 2+ hours. Check periodically:**

Every 30 minutes:
```bash
cd ~/distributed_prng_analysis

echo "=== 30-MIN CHECK $(date) ==="

# Memory
WATCHER_PID=$(pgrep -f "watcher_agent" | head -1)
ps -o pid,rss,%mem,etime -p "$WATCHER_PID"

# GPU VRAM (should be low between dispatches)
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Errors
tail -5 logs/watcher_agent.log | grep -i "error"

# Heuristic fallback count (should be 0 — real LLM should handle everything)
grep -c "heuristic_fallback" watcher_decisions.jsonl
```

**Step 5 — Stop and analyze:**

```bash
python3 -m agents.watcher_agent --halt "Soak Test C complete"
# Stop injector (Ctrl+C in Terminal 3)
```

**Post-test analysis:**

```bash
cd ~/distributed_prng_analysis

echo "=== SOAK TEST C RESULTS ==="
echo ""

# 1. Draws ingested during test
DRAWS_NOW=$(python3 -c "import json; d=json.load(open('lottery_history.json')); print(len(d['draws']))")
DRAWS_START=$(cat logs/soak_C_start_draws.txt 2>/dev/null || echo "unknown")
echo "Draws: $DRAWS_START → $DRAWS_NOW"

# 2. Cycles completed
CYCLES=$(grep -c "COMPLETED\|completed" watcher_decisions.jsonl 2>/dev/null || echo 0)
echo "Decision cycles: $CYCLES"

# 3. Dispatch counts
SELFPLAY_COUNT=$(grep -c "dispatch_selfplay" watcher_decisions.jsonl 2>/dev/null || echo 0)
LEARN_COUNT=$(grep -c "dispatch_learning" watcher_decisions.jsonl 2>/dev/null || echo 0)
echo "Selfplay dispatches: $SELFPLAY_COUNT"
echo "Learning loop dispatches: $LEARN_COUNT"

# 4. Heuristic fallbacks (should be 0)
HEURISTIC=$(grep -c "heuristic" watcher_decisions.jsonl 2>/dev/null || echo 0)
echo "Heuristic fallbacks: $HEURISTIC (target: 0)"

# 5. Errors
ERRORS=$(grep -ci "error\|exception\|traceback" logs/soak_C_*.log 2>/dev/null || echo 0)
echo "Total errors in log: $ERRORS"

# 6. Memory delta
echo ""
echo "Final process state:"
ps -o pid,rss,vsz,%mem,etime -p "$(pgrep -f watcher_agent | head -1)" 2>/dev/null || echo "Process already exited"

# 7. Policy candidates generated
echo ""
echo "Policy candidates: $(ls policy_history/ 2>/dev/null | wc -l)"

# 8. Telemetry health
echo ""
echo "Telemetry:"
python3 -c "
import json
try:
    with open('telemetry/learning_health_latest.json') as f:
        t = json.load(f)
    print(f'  Models tracked: {t.get(\"total_models\", \"N/A\")}')
    print(f'  Last updated: {t.get(\"timestamp\", \"N/A\")}')
except Exception as e:
    print(f'  Error reading telemetry: {e}')
"

# 9. Convergence signal (key metric for known-seed test)
echo ""
echo "Convergence check:"
python3 -c "
import json
try:
    with open('prediction_pool.json') as f:
        pool = json.load(f)
    print(f'  Pool size: {len(pool.get(\"predictions\", []))}')
except:
    print('  No prediction pool yet (expected if learning loop hasnt completed)')
"

# Cleanup halt flag for next use
rm -f watcher_halt.flag
```

### Pass Criteria

| Metric | Pass | Fail |
|--------|------|------|
| Completed cycles | ≥ 5 over 2 hours | 0 or stuck |
| Draws ingested | Matches injection rate (± 2) | Draws missing or duplicated |
| Selfplay dispatches | ≥ 1 | 0 (never triggered) |
| Heuristic fallbacks | 0 | > 2 (LLM connection issues) |
| Memory growth | < 100 MB over 2 hours | > 300 MB (leak) |
| Errors | 0 unhandled exceptions | Any traceback |
| LLM lifecycle | Clean stop/restart every cycle | VRAM accumulation or hung server |
| Policy candidates | Accumulating in `policy_history/` | No new candidates |
| Telemetry | Valid JSON, model count increasing | Corrupt or stale |
| Request queue | All processed, none stuck | Stuck or lost requests |

---

## 7. Failure Triage Guide

### Memory Leak

**Symptom:** RSS grows monotonically > 50 MB/hour

**Diagnosis:**
```bash
# Identify what's growing
python3 -c "
import tracemalloc
tracemalloc.start()
# ... after running for a while
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)
"
```

**Common causes:**
- Decision history appended to in-memory list (should be file-only)
- LLM response buffers not freed
- Survivor data loaded but never GC'd

### LLM Connection Drop

**Symptom:** `heuristic_fallback` entries in decision log, HTTP 000 in monitor

**Diagnosis:**
```bash
# Check LLM server logs
tail -50 logs/llm_server.log

# Verify port is still bound
ss -tlnp | grep 8080

# Check GPU VRAM (model may have been evicted)
nvidia-smi
```

**Fix:** The lifecycle manager should auto-restart. If not, manual restart:
```bash
bash llm_services/start_llm_servers.sh
```

### Queue Corruption

**Symptom:** Requests stuck pending, never archived

**Diagnosis:**
```bash
# Check request file permissions
ls -la watcher_requests/*.json

# Try parsing each request
for f in watcher_requests/*.json; do
    python3 -c "import json; json.load(open('$f'))" && echo "OK: $f" || echo "CORRUPT: $f"
done
```

### VRAM Not Freed

**Symptom:** GPU memory stays high between dispatches

**Diagnosis:**
```bash
# Check what's holding VRAM
nvidia-smi
fuser /dev/nvidia*  # Show PIDs using GPU

# Force cleanup
python3 -c "import torch; torch.cuda.empty_cache()"
```

**Root cause usually:** `llm_lifecycle.stop()` not called, or subprocess not fully terminated.

### Disk Space Exhaustion

**Symptom:** Write errors in log, df shows < 1 GB

**Diagnosis:**
```bash
du -sh logs/ watcher_requests/archive/ policy_history/ telemetry/ diagnostics_history/

# Clean old archives if needed
find watcher_requests/archive/ -name "*.json" -mtime +7 -delete
find logs/ -name "*.log" -mtime +3 -delete
```

---

## 8. Execution Sequence

Recommended order (each test builds confidence for the next):

```
Soak A (daemon endurance)          ← Proves daemon survives
    ↓
Soak B (sequential requests)       ← Proves dispatch + queue integrity
    ↓
Soak C (full autonomous loop)      ← Proves the complete system
```

**Total estimated time:** 5-8 hours (can span multiple sessions)

Allow cooldown between tests:
```bash
# Between tests — reset state
rm -f watcher_halt.flag
pkill -f "watcher_agent" 2>/dev/null
pkill -f "synthetic_draw_injector" 2>/dev/null
sleep 10

# Verify clean
pgrep -f "watcher_agent" && echo "STILL RUNNING" || echo "Clean"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

---

## 9. Post-Soak Documentation

After all three tests pass, create a session changelog entry:

```
SESSION_CHANGELOG_20260203_SOAK.md

Soak Test A: PASS/FAIL — [duration], [memory delta], [notes]
Soak Test B: PASS/FAIL — [requests processed], [errors], [notes]
Soak Test C: PASS/FAIL — [cycles], [draws], [convergence signal], [notes]

Verdict: Autonomous loop [PRODUCTION-READY / NEEDS FIXES]
```

Update:
- `CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_0.md` — add soak test results
- `TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md` — mark soak testing complete

---

## 10. What Comes After Soak Testing

Once all three soak tests pass:

1. **Track 2: Bundle Factory Tier Enhancement** — Fill the three stub functions in `bundle_factory.py` (Tier 2 retrieval) for richer LLM context
2. **Phase 9B.3: Policy Proposal Heuristics** — Automatic heuristic generation (deferred until 9B.2 validated by soak tests)
3. **Parameter Advisor (Item B)** — LLM-advised parameter tuning for Steps 4-6
4. **`--save-all-models` flag** — Save all 4 Step 5 models for post-hoc AI analysis
5. **Production deployment** — Switch from synthetic injection to real draw monitoring

---

*End of Soak Test Plan v1.0.0*
