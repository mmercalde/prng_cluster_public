#!/usr/bin/env python3
"""
Progress Monitor - Rich Terminal Display
=========================================
Run this in a separate terminal to see beautiful progress display.

Usage:
    python3 progress_monitor.py
"""

import json
import time
import sys
from pathlib import Path
from datetime import timedelta

PROGRESS_FILE = "/tmp/cluster_progress.json"

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


def read_progress():
    """Read progress from JSON file"""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def generate_display(state: dict) -> Panel:
    """Generate rich display panel from state"""
    if state is None:
        return Panel("[yellow]Waiting for cluster activity...[/yellow]\n\nStart a workflow to see progress here.", 
                     title="🔍 Cluster Monitor", border_style="yellow")
    
    step_name = state.get("step_name", "Unknown")
    total_jobs = state.get("total_jobs", 100)
    jobs_completed = state.get("jobs_completed", 0)
    seeds_completed = state.get("seeds_completed", 0)
    elapsed = state.get("elapsed_seconds", 0)
    nodes = state.get("nodes", {})
    finished = state.get("finished", False)
    
    # Calculate progress based on seeds (more accurate)
    total_seeds = state.get("total_seeds", total_jobs * 100000)  # Estimate if not set
    progress_pct = (seeds_completed / total_seeds * 100) if total_seeds > 0 else 0
    total_sps = sum(n.get("current_seeds_per_sec", 0) for n in nodes.values())
    
    # ETA calculation based on throughput
    if total_sps > 0 and not finished:
        remaining_seeds = total_seeds - seeds_completed
        eta = remaining_seeds / total_sps
        eta_str = str(timedelta(seconds=int(eta)))
    elif finished:
        eta_str = "Complete!"
    elif seeds_completed > 0:
        # Fallback to time-based estimate
        remaining_seeds = total_seeds - seeds_completed
        eta = (elapsed / seeds_completed) * remaining_seeds
        eta_str = str(timedelta(seconds=int(eta)))
    else:
        eta_str = "calculating..."
    
    # Progress bar
    bar_width = 40
    filled = int(bar_width * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    
    # Build output
    lines = []
    lines.append(f"[bold cyan]{step_name}[/bold cyan]")
    lines.append("")
    
    if finished:
        lines.append(f"[green][{bar}] {progress_pct:5.1f}% ✅ COMPLETE[/green]")
    else:
        lines.append(f"[{bar}] {progress_pct:5.1f}%")
    
    total_seeds = state.get("total_seeds", total_jobs * 100000)
    lines.append(f"Seeds: {seeds_completed:,}/{total_seeds:,} | Jobs: {jobs_completed}/{total_jobs} | ETA: {eta_str}")
    lines.append(f"Cluster throughput: [green]{total_sps:,.0f}[/green] seeds/sec | Elapsed: {timedelta(seconds=int(elapsed))}")
    lines.append("")
    lines.append("[bold]Nodes:[/bold]")
    
    # Node status
    for hostname, node in sorted(nodes.items()):
        total_gpus = node.get("total_gpus", 0)
        active_gpus = min(node.get("jobs_completed", 0), total_gpus)  # Approximation
        gpu_type = node.get("gpu_type", "Unknown")
        sps = node.get("current_seeds_per_sec", 0)
        jobs = node.get("jobs_completed", 0)
        
        # GPU bar
        if total_gpus > 0:
            gpu_bar = "█" * min(active_gpus, total_gpus) + "░" * max(0, total_gpus - active_gpus)
        else:
            gpu_bar = "????"
        
        # Shorten hostname
        display_host = hostname if len(hostname) <= 15 else hostname[:12] + "..."
        
        # Color based on activity
        sps_color = "green" if sps > 0 else "dim"
        
        lines.append(
            f"  {display_host:15} [{gpu_bar}] "
            f"[{sps_color}]{sps:6,.0f}[/{sps_color}] seeds/s | "
            f"{jobs:,} jobs | {gpu_type}"
        )
    
    border_color = "green" if finished else "blue"
    return Panel("\n".join(lines), border_style=border_color)


def main():
    console = Console()
    
    console.print("[bold cyan]🖥️  Cluster Progress Monitor[/bold cyan]")
    console.print(f"Watching: {PROGRESS_FILE}")
    console.print("Press Ctrl+C to exit\n")
    
    last_update = 0
    
    with Live(generate_display(None), console=console, refresh_per_second=2) as live:
        try:
            while True:
                state = read_progress()
                
                # Check if state updated
                if state:
                    updated_at = state.get("updated_at", 0)
                    if updated_at != last_update:
                        last_update = updated_at
                        live.update(generate_display(state))
                    
                    # Exit if finished and no updates for 5 seconds
                    if state.get("finished") and (time.time() - updated_at) > 5:
                        live.update(generate_display(state))
                        break
                
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitor stopped.[/yellow]")


if __name__ == "__main__":
    main()
