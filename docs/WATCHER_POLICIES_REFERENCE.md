# WATCHER POLICIES REFERENCE
## Complete Flag Documentation for `watcher_policies.json`

**Version:** 1.0.0  
**Date:** 2026-02-11 (Session 80)  
**Authoritative Source:** This document defines the canonical meaning of every policy flag.

---

## Quick Reference: Production vs Test Mode

```bash
# CHECK current state:
python3 -c "
import json
with open('watcher_policies.json') as f: p = json.load(f)
for k in ['test_mode','auto_approve_in_test_mode','skip_escalation_in_test_mode','approval_route']:
    print(f'  {k}: {p.get(k, \"NOT SET\")}')
"

# SET production mode:
python3 -c "
import json
with open('watcher_policies.json') as f: p = json.load(f)
p['test_mode'] = False
p['auto_approve_in_test_mode'] = False
p['skip_escalation_in_test_mode'] = False
p['approval_route'] = 'orchestrator'
with open('watcher_policies.json','w') as f: json.dump(p,f,indent=2)
print('Production mode set')
"

# SET test/soak mode:
python3 -c "
import json
with open('watcher_policies.json') as f: p = json.load(f)
p['test_mode'] = True
p['auto_approve_in_test_mode'] = True
p['skip_escalation_in_test_mode'] = True
p['approval_route'] = 'watcher'
with open('watcher_policies.json','w') as f: json.dump(p,f,indent=2)
print('Test/soak mode set')
"
```

---

## Flag Definitions

### `test_mode`
| | |
|---|---|
| **Type** | boolean |
| **Default** | `false` |
| **Production** | `false` |
| **Test/Soak** | `true` |
| **Purpose** | Master switch for test mode. Enables synthetic draw injection and allows other test-mode flags to take effect. |
| **When true** | Synthetic draws accepted. Safety gates relaxed. Delta rejection can be skipped. |
| **When false** | Only real draws accepted. All safety gates active. |
| **Set by** | Human operator only |
| **Read by** | `synthetic_draw_injector.py`, `chapter_13_acceptance.py`, `chapter_13_orchestrator.py`, `watcher_agent.py` |

---

### `auto_approve_in_test_mode`
| | |
|---|---|
| **Type** | boolean |
| **Default** | `false` |
| **Production** | `false` |
| **Test/Soak** | `true` |
| **Purpose** | When combined with `test_mode=true`, allows WATCHER daemon or orchestrator to auto-approve retrain proposals without human confirmation. |
| **Requires** | `test_mode = true` (both flags checked) |
| **When true + test_mode** | WATCHER auto-approves `pending_approval.json`. Orchestrator auto-executes (if `approval_route=orchestrator`). |
| **When false** | All approvals require human `--approve` CLI command. |
| **Set by** | Human operator only |
| **Read by** | `watcher_agent.py` (`_poll_pending_approval`), `chapter_13_orchestrator.py`, `chapter_13_acceptance.py` |

---

### `skip_escalation_in_test_mode`
| | |
|---|---|
| **Type** | boolean |
| **Default** | `false` |
| **Production** | `false` |
| **Test/Soak** | `true` |
| **Purpose** | Suppresses mandatory escalation in the acceptance engine during test mode. Without this, synthetic data (0% hit rate) always triggers escalation due to consecutive misses. |
| **Requires** | `test_mode = true` |
| **When true + test_mode** | Escalation reasons are cleared. Proposals proceed to accept/reject instead of escalate. |
| **When false** | Normal escalation logic applies. |
| **Set by** | Human operator only |
| **Read by** | `chapter_13_acceptance.py` |

---

### `approval_route`
| | |
|---|---|
| **Type** | string enum: `"orchestrator"` or `"watcher"` |
| **Default** | `"orchestrator"` |
| **Production** | `"orchestrator"` |
| **Test/Soak** | `"watcher"` |
| **Purpose** | Determines who owns execution authority after a proposal is accepted. |
| **`"orchestrator"`** | Chapter 13 orchestrator auto-executes internally (legacy behavior). Never creates `pending_approval.json` when auto-approving. |
| **`"watcher"`** | Orchestrator creates `pending_approval.json`. WATCHER daemon detects, approves, and dispatches `run_pipeline()`. |
| **Invalid values** | Silently fall back to `"orchestrator"` (enum validation in orchestrator). |
| **Set by** | Human operator only |
| **Read by** | `chapter_13_orchestrator.py` (line ~337) |

**Authority model:**
```
approval_route = "orchestrator"  →  Ch13 decides AND executes
approval_route = "watcher"       →  Ch13 decides, WATCHER executes
```

---

### `synthetic_injection.enabled`
| | |
|---|---|
| **Type** | boolean (nested under `synthetic_injection`) |
| **Default** | `false` |
| **Production** | `false` |
| **Test/Soak** | `true` |
| **Purpose** | Allows synthetic draw injector to generate and append draws. |
| **Requires** | `test_mode = true` (injector checks both) |
| **Read by** | `synthetic_draw_injector.py` |

---

### `synthetic_injection.interval_seconds`
| | |
|---|---|
| **Type** | integer |
| **Default** | `120` |
| **Test/Soak** | `60` (faster cycles) |
| **Purpose** | Seconds between synthetic draw injections in daemon mode. |
| **Read by** | `synthetic_draw_injector.py --daemon` |

---

### `synthetic_injection.true_seed`
| | |
|---|---|
| **Type** | integer |
| **Default** | `12345` |
| **Purpose** | Known seed for synthetic PRNG. Allows convergence measurement. |
| **Read by** | `synthetic_draw_injector.py` |

---

## Common Configurations

### Production (Normal Operation)
```json
{
  "test_mode": false,
  "auto_approve_in_test_mode": false,
  "skip_escalation_in_test_mode": false,
  "approval_route": "orchestrator"
}
```
Human approves all retrain requests via `--approve`.

### Soak Test (Full Autonomy)
```json
{
  "test_mode": true,
  "auto_approve_in_test_mode": true,
  "skip_escalation_in_test_mode": true,
  "approval_route": "watcher"
}
```
Fully autonomous. WATCHER daemon handles all approvals.

### Soak Test (Orchestrator Route)
```json
{
  "test_mode": true,
  "auto_approve_in_test_mode": true,
  "skip_escalation_in_test_mode": true,
  "approval_route": "orchestrator"
}
```
Orchestrator auto-executes internally. WATCHER not involved.

### Manual Testing (Human Approval)
```json
{
  "test_mode": true,
  "auto_approve_in_test_mode": false,
  "skip_escalation_in_test_mode": true,
  "approval_route": "watcher"
}
```
Orchestrator creates `pending_approval.json`. Human runs `watcher_agent.py --approve`.

---

## Safety Invariants

1. **`test_mode=false` overrides everything** — No auto-approve, no synthetic injection, no escalation skip regardless of other flags.
2. **Auto-approve requires BOTH flags** — `test_mode=true` AND `auto_approve_in_test_mode=true`. Either one alone does nothing.
3. **`approval_route` is governance** — Determines execution authority, not approval logic.
4. **Invalid enum values fail safe** — Unknown `approval_route` values silently revert to `"orchestrator"`.
5. **WATCHER never mutates policies** — Only humans change `watcher_policies.json`.

---

*End of Policy Reference*
