#!/usr/bin/env python3
"""
Progress Monitor - Rich Terminal Display
=========================================
Run this in a separate terminal to see beautiful progress display.

v2 enhancements:
  - Continuous mode: survives trial boundaries — never exits between phases
  - Trial N of M header above progress bar (reads pipeline log directly)
  - --log-pattern arg to target the active run log
  - "Waiting for next phase" panel between trials instead of exit

Usage:
    python3 progress_monitor.py
    python3 progress_monitor.py --log-pattern s163_karg_gdm_clean
    python3 progress_monitor.py --trials 5
"""

import argparse
import glob
import json
import os
import re
import subprocess
import time
import sys
from pathlib import Path
from datetime import timedelta

PROGRESS_FILE = "/tmp/cluster_progress.json"
LOG_DIR = os.path.expanduser("~/distributed_prng_analysis/logs")

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("❌ 'rich' library required. Install with:")
    print("   pip install rich --break-system-packages")
    sys.exit(1)


def find_active_log(pattern: str | None) -> str | None:
    """Find the most recent matching pipeline log on Zeus (local path)."""
    if pattern:
        candidates = sorted(glob.glob(f"{LOG_DIR}/{pattern}*.log"), key=os.path.getmtime)
    else:
        candidates = sorted(glob.glob(f"{LOG_DIR}/s*.log"), key=os.path.getmtime)
    # Exclude dashboard/netconsole/soak logs
    candidates = [c for c in candidates
                  if not any(x in os.path.basename(c)
                             for x in ("dashboard", "netconsole", "soak"))]
    return candidates[-1] if candidates else None


def get_trial_context(log_file: str | None, total_trials: int) -> dict:
    """
    Parse the pipeline log for current trial number, config name, and phase.
    Returns dict: {trial_num, trial_config, total_trials, phase}
    """
    ctx = {"trial_num": None, "trial_config": None,
           "total_trials": total_trials, "phase": None}
    if not log_file or not os.path.exists(log_file):
        return ctx
    try:
        # Read last 300 lines — trial markers appear here
        result = subprocess.run(
            ["tail", "-300", log_file],
            capture_output=True, text=True, timeout=3
        )
        lines = result.stdout.splitlines()

        # Find last "Trial N:" line  e.g. "Trial 3: W10_O61_midday_S1-243 → Score: 0.00"
        # or "✨ NEW BEST [Trial N]:"
        trial_num = None
        trial_config = None
        for line in reversed(lines):
            m = re.search(r'Trial\s+(\d+)[:\s]', line)
            if m:
                trial_num = int(m.group(1))
                # Try to extract config name W\d+_O\d+_...
                cm = re.search(r'(W\d+_O\d+_\S+)', line)
                if cm:
                    trial_config = cm.group(1).rstrip(".,")
                break

        # Find current phase from progress file or log
        for line in reversed(lines):
            if "forward sieve" in line.lower() or "forward_sieve" in line.lower():
                ctx["phase"] = "forward"
                break
            elif "reverse sieve" in line.lower() or "reverse_sieve" in line.lower():
                ctx["phase"] = "reverse"
                break
            elif "bidirectional" in line.lower():
                ctx["phase"] = "bidirectional"
                break
            elif "hybrid" in line.lower():
                ctx["phase"] = "hybrid"
                break

        ctx["trial_num"]    = trial_num
        ctx["trial_config"] = trial_config
    except Exception:
        pass
    return ctx


def read_progress() -> dict | None:
    """Read progress from JSON file."""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def generate_display(state: dict | None, trial_ctx: dict | None = None) -> Panel:
    """Generate rich display panel from state."""
    if state is None:
        return Panel(
            "[yellow]Waiting for cluster activity...[/yellow]\n\n"
            "Start a workflow to see progress here.",
            title="🔍 Cluster Monitor", border_style="yellow"
        )

    step_name      = state.get("step_name", "Unknown")
    total_jobs     = state.get("total_jobs", 100)
    jobs_completed = state.get("jobs_completed", 0)
    seeds_completed = state.get("seeds_completed", 0)
    elapsed        = state.get("elapsed_seconds", 0)
    nodes          = state.get("nodes", {})
    finished       = state.get("finished", False)
    total_seeds    = state.get("total_seeds", total_jobs * 100000)

    progress_pct = (seeds_completed / total_seeds * 100) if total_seeds > 0 else 0
    total_sps    = sum(n.get("current_seeds_per_sec", 0) for n in nodes.values())

    if total_sps > 0 and not finished:
        remaining_seeds = total_seeds - seeds_completed
        eta_str = str(timedelta(seconds=int(remaining_seeds / total_sps)))
    elif finished:
        eta_str = "Complete!"
    elif seeds_completed > 0:
        remaining_seeds = total_seeds - seeds_completed
        eta_str = str(timedelta(seconds=int((elapsed / seeds_completed) * remaining_seeds)))
    else:
        eta_str = "calculating..."

    bar_width = 40
    filled = int(bar_width * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    lines = []

    # ── Trial header ──────────────────────────────────────────────────────────
    if trial_ctx:
        tn    = trial_ctx.get("trial_num")
        tc    = trial_ctx.get("trial_config")
        tt    = trial_ctx.get("total_trials", "?")
        phase = trial_ctx.get("phase")
        if tn:
            trial_str = f"Trial {tn}/{tt}"
            if tc:
                trial_str += f"  [{tc}]"
            if phase:
                trial_str += f"  phase={phase}"
            lines.append(f"[bold yellow]▶ {trial_str}[/bold yellow]")
            lines.append("")

    # ── Phase progress ────────────────────────────────────────────────────────
    lines.append(f"[bold cyan]{step_name}[/bold cyan]")
    lines.append("")

    if finished:
        lines.append(f"[green][{bar}] {progress_pct:5.1f}% ✅ COMPLETE[/green]")
    else:
        lines.append(f"[{bar}] {progress_pct:5.1f}%")

    lines.append(
        f"Seeds: {seeds_completed:,}/{total_seeds:,} | "
        f"Jobs: {jobs_completed}/{total_jobs} | ETA: {eta_str}"
    )
    lines.append(
        f"Cluster throughput: [green]{total_sps:,.0f}[/green] seeds/sec | "
        f"Elapsed: {timedelta(seconds=int(elapsed))}"
    )
    lines.append("")
    lines.append("[bold]Nodes:[/bold]")

    for hostname, node in sorted(nodes.items()):
        total_gpus  = node.get("total_gpus", 0)
        active_gpus = min(node.get("jobs_completed", 0), total_gpus)
        gpu_type    = node.get("gpu_type", "Unknown")
        sps         = node.get("current_seeds_per_sec", 0)
        jobs        = node.get("jobs_completed", 0)

        if total_gpus > 0:
            gpu_bar = "█" * min(active_gpus, total_gpus) + "░" * max(0, total_gpus - active_gpus)
        else:
            gpu_bar = "????"

        display_host = hostname if len(hostname) <= 15 else hostname[:12] + "..."
        sps_color    = "green" if sps > 0 else "dim"

        lines.append(
            f"  {display_host:15} [{gpu_bar}] "
            f"[{sps_color}]{sps:6,.0f}[/{sps_color}] seeds/s | "
            f"{jobs:,} jobs | {gpu_type}"
        )

    border_color = "green" if finished else "blue"
    return Panel("\n".join(lines), border_style=border_color)


def generate_waiting_panel(last_state: dict | None, trial_ctx: dict | None) -> Panel:
    """Panel shown between phases while waiting for next phase to start."""
    lines = []

    if trial_ctx and trial_ctx.get("trial_num"):
        tn = trial_ctx["trial_num"]
        tt = trial_ctx.get("total_trials", "?")
        lines.append(f"[bold yellow]▶ Trial {tn}/{tt} phase complete[/bold yellow]")
        lines.append("")

    lines.append("[green]✅ Phase complete[/green]")
    lines.append("")
    lines.append("[yellow]Waiting for next phase to start...[/yellow]")
    lines.append("[dim](cluster_progress.json will update when next phase begins)[/dim]")
    lines.append("")

    if last_state:
        step  = last_state.get("step_name", "?")
        seeds = last_state.get("seeds_completed", 0)
        sps   = sum(n.get("current_seeds_per_sec", 0)
                    for n in last_state.get("nodes", {}).values())
        elapsed = last_state.get("elapsed_seconds", 0)
        lines.append(f"[dim]Last phase: {step}[/dim]")
        lines.append(f"[dim]Seeds processed: {seeds:,}  |  "
                     f"Avg throughput: {sps:,.0f} s/s  |  "
                     f"Elapsed: {timedelta(seconds=int(elapsed))}[/dim]")

    return Panel("\n".join(lines),
                 title="⏳ Between Phases", border_style="yellow")


def main():
    ap = argparse.ArgumentParser(description="Cluster Progress Monitor v2 — continuous mode")
    ap.add_argument("--log-pattern", default=None,
                    help="Pipeline log filename prefix (e.g. s163_karg_gdm_clean)")
    ap.add_argument("--trials", type=int, default=5,
                    help="Total number of Optuna trials in this run (default: 5)")
    args = ap.parse_args()

    console = Console()
    console.print("[bold cyan]🖥️  Cluster Progress Monitor v2 — continuous mode[/bold cyan]")
    console.print(f"Watching: {PROGRESS_FILE}")
    if args.log_pattern:
        console.print(f"Log pattern: {args.log_pattern}")
    console.print(f"Total trials: {args.trials}")
    console.print("Press Ctrl+C to exit\n")

    log_file  = find_active_log(args.log_pattern)
    last_update = 0
    last_state  = None
    last_finished_at = 0

    with Live(generate_display(None), console=console, refresh_per_second=2) as live:
        try:
            while True:
                # Refresh log file path each cycle — new log may appear mid-run
                log_file = find_active_log(args.log_pattern) or log_file

                # Get trial context from pipeline log
                trial_ctx = get_trial_context(log_file, args.trials)

                state = read_progress()

                if state:
                    updated_at = state.get("updated_at", 0)

                    if updated_at != last_update:
                        last_update   = updated_at
                        last_state    = state
                        last_finished_at = 0  # reset — new data arriving
                        live.update(generate_display(state, trial_ctx))

                    if state.get("finished"):
                        if last_finished_at == 0:
                            last_finished_at = time.time()

                        # Give 5s of showing COMPLETE before switching to waiting panel
                        if time.time() - last_finished_at > 5:
                            # Show waiting panel — do NOT exit
                            live.update(generate_waiting_panel(last_state, trial_ctx))

                            # Wait for progress file to reset (next phase starts)
                            wait_start = time.time()
                            while True:
                                time.sleep(1)
                                new_state = read_progress()
                                if new_state and not new_state.get("finished", True):
                                    # New phase started
                                    last_update      = 0
                                    last_finished_at = 0
                                    break
                                # Keep updating trial context while waiting
                                trial_ctx = get_trial_context(log_file, args.trials)
                                live.update(generate_waiting_panel(last_state, trial_ctx))

                                # Safety: if waiting more than 10 min, show timeout note
                                if time.time() - wait_start > 600:
                                    live.update(Panel(
                                        "[yellow]No new phase in 10 minutes.\n"
                                        "Run may have completed all trials or coordinator stopped.[/yellow]",
                                        title="⏳ Waiting", border_style="dim"
                                    ))
                    else:
                        last_finished_at = 0

                else:
                    # No progress file — show waiting
                    live.update(generate_waiting_panel(last_state, trial_ctx)
                                if last_state else generate_display(None))

                time.sleep(0.5)

        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor stopped.[/yellow]")


if __name__ == "__main__":
    main()
