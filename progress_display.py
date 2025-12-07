#!/usr/bin/env python3
"""
Centralized Progress Display Module for PRNG Cluster Operations
================================================================
Version: 1.0.0

Provides clean, unified progress display across all pipeline steps.
Uses 'rich' library for beautiful terminal output, with fallback to tqdm/print.

Usage:
    from progress_display import PipelineProgress, ClusterProgress
    
    # For pipeline step progress
    with PipelineProgress("Step 1: Window Optimizer", total_trials=20) as progress:
        for i in range(20):
            progress.update(trials_complete=i+1, best_score=0.85)
    
    # For cluster GPU progress
    with ClusterProgress("Scoring", total_jobs=1000) as progress:
        progress.update_gpu("zeus", 0, jobs_done=50, seeds_per_sec=25000)
        progress.complete_job("job_001")
"""

import sys
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Try to import rich, fall back gracefully
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
    # Disable rich if stdout is not a TTY (e.g., captured by subprocess)
    if not sys.stdout.isatty():
        RICH_AVAILABLE = False
except ImportError:
    RICH_AVAILABLE = False

# Try tqdm as secondary fallback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class GPUStatus:
    """Status of a single GPU."""
    hostname: str
    gpu_id: int
    gpu_type: str = "Unknown"
    jobs_done: int = 0
    seeds_per_sec: float = 0.0
    last_update: float = field(default_factory=time.time)
    active: bool = True


@dataclass 
class PipelineStepStatus:
    """Status of a pipeline step."""
    step_num: int
    step_name: str
    status: str = "pending"  # pending, running, complete, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    trials_complete: int = 0
    total_trials: int = 0
    best_score: float = 0.0
    message: str = ""


class SimpleProgress:
    """Fallback progress display using simple prints."""
    
    def __init__(self, title: str, total: int = 100):
        self.title = title
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.last_print = 0
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def update(self, current: int = None, message: str = ""):
        if current is not None:
            self.current = current
        
        # Only print every 2 seconds or on completion
        now = time.time()
        if now - self.last_print < 2 and self.current < self.total:
            return
        
        self.last_print = now
        elapsed = now - self.start_time
        pct = (self.current / self.total * 100) if self.total > 0 else 0
        
        # Calculate ETA
        if self.current > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f"ETA: {timedelta(seconds=int(remaining))}"
        else:
            eta_str = "ETA: --:--"
        
        # Simple progress bar
        bar_width = 30
        filled = int(bar_width * self.current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        msg = f" | {message}" if message else ""
        print(f"\r  [{bar}] {pct:5.1f}% | {self.current}/{self.total} | {eta_str}{msg}    ", end="", flush=True)
        
        if self.current >= self.total:
            print()  # Newline on completion
    
    def complete(self, message: str = "Complete"):
        self.current = self.total
        elapsed = time.time() - self.start_time
        print(f"\n  ✅ {message} in {timedelta(seconds=int(elapsed))}")
        print(f"{'='*60}\n")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class ClusterProgress:
    """
    Progress display for distributed cluster operations.
    
    Shows:
    - Overall job progress
    - Per-node/GPU status
    - Throughput metrics
    """
    
    def __init__(
        self, 
        title: str,
        total_jobs: int,
        nodes: List[str] = None
    ):
        self.title = title
        self.total_jobs = total_jobs
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.start_time = time.time()
        self.gpu_status: Dict[str, GPUStatus] = {}
        self.nodes = nodes or []
        self.console = Console() if RICH_AVAILABLE else None
        self.live = None
        self._last_simple_update = 0
    
    def _make_table(self) -> "Table":
        """Create rich table showing cluster status."""
        table = Table(title=self.title, show_header=True)
        table.add_column("Node", style="cyan")
        table.add_column("GPUs", justify="center")
        table.add_column("Jobs Done", justify="right")
        table.add_column("Speed", justify="right", style="green")
        table.add_column("Status", justify="center")
        
        # Group GPUs by node
        nodes_gpus: Dict[str, List[GPUStatus]] = {}
        for key, gpu in self.gpu_status.items():
            if gpu.hostname not in nodes_gpus:
                nodes_gpus[gpu.hostname] = []
            nodes_gpus[gpu.hostname].append(gpu)
        
        for hostname, gpus in sorted(nodes_gpus.items()):
            active = sum(1 for g in gpus if g.active)
            total_gpus = len(gpus)
            jobs = sum(g.jobs_done for g in gpus)
            speed = sum(g.seeds_per_sec for g in gpus)
            
            status = "🟢" if active == total_gpus else "🟡" if active > 0 else "🔴"
            
            table.add_row(
                hostname,
                f"{active}/{total_gpus}",
                str(jobs),
                f"{speed/1000:.1f}K/s" if speed > 0 else "-",
                status
            )
        
        return table
    
    def _make_progress_panel(self) -> "Panel":
        """Create progress panel."""
        elapsed = time.time() - self.start_time
        pct = (self.completed_jobs / self.total_jobs * 100) if self.total_jobs > 0 else 0
        
        # Calculate ETA
        if self.completed_jobs > 0:
            rate = self.completed_jobs / elapsed
            remaining = (self.total_jobs - self.completed_jobs) / rate if rate > 0 else 0
            eta = timedelta(seconds=int(remaining))
        else:
            eta = "--:--"
        
        # Progress bar
        bar_width = 40
        filled = int(bar_width * self.completed_jobs / self.total_jobs) if self.total_jobs > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        text = Text()
        text.append(f"[{bar}] {pct:.1f}%\n", style="bold")
        text.append(f"Jobs: {self.completed_jobs}/{self.total_jobs}", style="cyan")
        if self.failed_jobs > 0:
            text.append(f" | Failed: {self.failed_jobs}", style="red")
        text.append(f"\nElapsed: {timedelta(seconds=int(elapsed))} | ETA: {eta}")
        
        return Panel(text, title="Progress", border_style="green")
    
    def _render(self) -> "Layout":
        """Render full display."""
        layout = Layout()
        layout.split_column(
            Layout(self._make_progress_panel(), size=6),
            Layout(self._make_table())
        )
        return layout
    
    def update_gpu(
        self, 
        hostname: str, 
        gpu_id: int,
        jobs_done: int = None,
        seeds_per_sec: float = None,
        gpu_type: str = None,
        active: bool = True
    ):
        """Update status for a specific GPU."""
        key = f"{hostname}:{gpu_id}"
        
        if key not in self.gpu_status:
            self.gpu_status[key] = GPUStatus(
                hostname=hostname,
                gpu_id=gpu_id,
                gpu_type=gpu_type or "Unknown"
            )
        
        gpu = self.gpu_status[key]
        if jobs_done is not None:
            gpu.jobs_done = jobs_done
        if seeds_per_sec is not None:
            gpu.seeds_per_sec = seeds_per_sec
        if gpu_type is not None:
            gpu.gpu_type = gpu_type
        gpu.active = active
        gpu.last_update = time.time()
        
        if self.live:
            self.live.update(self._render())
    
    def complete_job(self, job_id: str = None, success: bool = True):
        """Mark a job as complete."""
        if success:
            self.completed_jobs += 1
        else:
            self.failed_jobs += 1
        
        if self.live:
            self.live.update(self._render())
        elif not RICH_AVAILABLE:
            # Simple fallback - print every 2 seconds
            now = time.time()
            if now - self._last_simple_update >= 2:
                self._last_simple_update = now
                pct = (self.completed_jobs / self.total_jobs * 100) if self.total_jobs > 0 else 0
                print(f"\r  Progress: {pct:.1f}% | {self.completed_jobs}/{self.total_jobs} jobs    ", end="", flush=True)
    
    def __enter__(self):
        if RICH_AVAILABLE and sys.stdout.isatty():
            self.live = Live(self._render(), console=self.console, refresh_per_second=2)
            self.live.__enter__()
        else:
            print(f"\n{'='*60}")
            print(f"  {self.title}")
            print(f"  Total jobs: {self.total_jobs}")
            print(f"{'='*60}")
        return self
    
    def __exit__(self, *args):
        if self.live:
            self.live.__exit__(*args)
        
        # Print summary
        elapsed = time.time() - self.start_time
        print(f"\n  ✅ Complete: {self.completed_jobs}/{self.total_jobs} jobs")
        if self.failed_jobs > 0:
            print(f"  ❌ Failed: {self.failed_jobs}")
        print(f"  ⏱️  Time: {timedelta(seconds=int(elapsed))}")
        print(f"{'='*60}\n")


class PipelineProgress:
    """
    Progress display for pipeline step execution.
    
    Shows:
    - Current step and trial progress
    - Best score so far
    - Timing estimates
    """
    
    def __init__(
        self,
        title: str,
        total_trials: int = 100,
        show_best_score: bool = True
    ):
        self.title = title
        self.total_trials = total_trials
        self.show_best_score = show_best_score
        self.trials_complete = 0
        self.best_score = 0.0
        self.best_config: Dict[str, Any] = {}
        self.start_time = time.time()
        self.console = Console() if RICH_AVAILABLE else None
        self.progress = None
        self.task_id = None
        self._simple = None
    
    def update(
        self,
        trials_complete: int = None,
        best_score: float = None,
        best_config: Dict[str, Any] = None,
        message: str = ""
    ):
        """Update progress."""
        if trials_complete is not None:
            self.trials_complete = trials_complete
        if best_score is not None:
            self.best_score = best_score
        if best_config is not None:
            self.best_config = best_config
        
        if self.progress and self.task_id is not None:
            desc = f"Best: {self.best_score:.4f}" if self.show_best_score else ""
            self.progress.update(self.task_id, completed=self.trials_complete, description=desc)
        elif self._simple:
            msg = f"Best: {self.best_score:.4f}" if self.show_best_score else message
            self._simple.update(self.trials_complete, msg)
    
    def log(self, message: str, style: str = ""):
        """Log a message below the progress bar."""
        if self.console and RICH_AVAILABLE:
            self.console.print(f"  {message}", style=style)
        else:
            print(f"  {message}")
    
    def __enter__(self):
        if RICH_AVAILABLE and sys.stdout.isatty():
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.__enter__()
            self.task_id = self.progress.add_task(self.title, total=self.total_trials)
        else:
            self._simple = SimpleProgress(self.title, self.total_trials)
        return self
    
    def __exit__(self, *args):
        if self.progress:
            self.progress.__exit__(*args)
        
        elapsed = time.time() - self.start_time
        print(f"\n  ✅ {self.title} complete")
        if self.show_best_score:
            print(f"  🏆 Best score: {self.best_score:.4f}")
        print(f"  ⏱️  Time: {timedelta(seconds=int(elapsed))}")


class WatcherProgress:
    """
    Progress display for Watcher Agent pipeline execution.
    
    Shows all 6 pipeline steps with status indicators.
    """
    
    STEP_NAMES = {
        1: "Window Optimizer",
        2: "Scorer Meta-Optimizer", 
        3: "Full Scoring",
        4: "ML Meta-Optimizer",
        5: "Anti-Overfit Training",
        6: "Prediction Generator"
    }
    
    def __init__(self):
        self.steps: Dict[int, PipelineStepStatus] = {}
        for i in range(1, 7):
            self.steps[i] = PipelineStepStatus(
                step_num=i,
                step_name=self.STEP_NAMES[i]
            )
        self.current_step = 0
        self.console = Console() if RICH_AVAILABLE else None
        self.live = None
    
    def _make_table(self) -> "Table":
        """Create pipeline status table."""
        table = Table(title="🔬 PRNG Analysis Pipeline", show_header=True)
        table.add_column("Step", justify="center", width=4)
        table.add_column("Name", width=25)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Progress", width=20)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Time", justify="right", width=10)
        
        status_icons = {
            "pending": "⬜",
            "running": "🔄",
            "complete": "✅",
            "failed": "❌",
            "skipped": "⏭️"
        }
        
        for i in range(1, 7):
            step = self.steps[i]
            icon = status_icons.get(step.status, "❓")
            
            # Progress text
            if step.status == "running" and step.total_trials > 0:
                pct = step.trials_complete / step.total_trials * 100
                progress = f"{step.trials_complete}/{step.total_trials} ({pct:.0f}%)"
            elif step.status == "complete":
                progress = "Done"
            else:
                progress = "-"
            
            # Score
            score = f"{step.best_score:.4f}" if step.best_score > 0 else "-"
            
            # Time
            if step.end_time and step.start_time:
                elapsed = step.end_time - step.start_time
                time_str = str(timedelta(seconds=int(elapsed)))
            elif step.start_time:
                elapsed = time.time() - step.start_time
                time_str = f"{timedelta(seconds=int(elapsed))}..."
            else:
                time_str = "-"
            
            # Highlight current step
            style = "bold yellow" if step.status == "running" else ""
            
            table.add_row(
                str(i),
                step.step_name,
                icon,
                progress,
                score,
                time_str,
                style=style
            )
        
        return table
    
    def start_step(self, step: int, total_trials: int = 0):
        """Mark a step as started."""
        self.current_step = step
        self.steps[step].status = "running"
        self.steps[step].start_time = time.time()
        self.steps[step].total_trials = total_trials
        self._refresh()
    
    def update_step(
        self,
        step: int,
        trials_complete: int = None,
        best_score: float = None,
        message: str = ""
    ):
        """Update step progress."""
        if trials_complete is not None:
            self.steps[step].trials_complete = trials_complete
        if best_score is not None:
            self.steps[step].best_score = best_score
        if message:
            self.steps[step].message = message
        self._refresh()
    
    def complete_step(self, step: int, success: bool = True, score: float = None):
        """Mark a step as complete."""
        self.steps[step].status = "complete" if success else "failed"
        self.steps[step].end_time = time.time()
        if score is not None:
            self.steps[step].best_score = score
        self._refresh()
    
    def _refresh(self):
        """Refresh the display."""
        if self.live:
            self.live.update(self._make_table())
        elif not RICH_AVAILABLE:
            # Simple text output
            step = self.steps.get(self.current_step)
            if step and step.status == "running":
                pct = (step.trials_complete / step.total_trials * 100) if step.total_trials > 0 else 0
                print(f"\r  Step {self.current_step}: {step.step_name} | {pct:.0f}% | Best: {step.best_score:.4f}    ", end="", flush=True)
    
    def __enter__(self):
        if RICH_AVAILABLE and sys.stdout.isatty():
            self.live = Live(self._make_table(), console=self.console, refresh_per_second=1)
            self.live.__enter__()
        else:
            print("\n" + "="*60)
            print("  🔬 PRNG Analysis Pipeline")
            print("="*60)
        return self
    
    def __exit__(self, *args):
        if self.live:
            self.live.__exit__(*args)
        
        # Print final summary
        print("\n" + "="*60)
        print("  Pipeline Summary:")
        for i in range(1, 7):
            step = self.steps[i]
            icon = "✅" if step.status == "complete" else "❌" if step.status == "failed" else "⬜"
            score = f" (score: {step.best_score:.4f})" if step.best_score > 0 else ""
            print(f"  {icon} Step {i}: {step.step_name}{score}")
        print("="*60 + "\n")


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def check_display_support() -> Dict[str, bool]:
    """Check what display libraries are available."""
    return {
        "rich": RICH_AVAILABLE,
        "tqdm": TQDM_AVAILABLE,
        "tty": sys.stdout.isatty()
    }


def print_banner(title: str, subtitle: str = ""):
    """Print a nice banner."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel(
            f"[bold]{title}[/bold]\n{subtitle}" if subtitle else f"[bold]{title}[/bold]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*60)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("="*60 + "\n")


# ════════════════════════════════════════════════════════════════════════════════
# TEST / DEMO
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Progress Display Module Test")
    print(f"Rich available: {RICH_AVAILABLE}")
    print(f"tqdm available: {TQDM_AVAILABLE}")
    print(f"TTY: {sys.stdout.isatty()}")
    
    # Test PipelineProgress
    print("\n--- Testing PipelineProgress ---")
    with PipelineProgress("Step 1: Window Optimizer", total_trials=10) as progress:
        for i in range(10):
            time.sleep(0.2)
            progress.update(trials_complete=i+1, best_score=0.5 + i*0.05)
    
    # Test ClusterProgress
    print("\n--- Testing ClusterProgress ---")
    with ClusterProgress("Distributed Scoring", total_jobs=20) as progress:
        for i in range(20):
            time.sleep(0.1)
            progress.update_gpu("zeus", 0, jobs_done=i//2, seeds_per_sec=25000)
            progress.update_gpu("rig-6600", 0, jobs_done=i//2, seeds_per_sec=5000)
            progress.complete_job(f"job_{i}")
    
    # Test WatcherProgress
    print("\n--- Testing WatcherProgress ---")
    with WatcherProgress() as watcher:
        for step in range(1, 4):
            watcher.start_step(step, total_trials=5)
            for t in range(5):
                time.sleep(0.1)
                watcher.update_step(step, trials_complete=t+1, best_score=0.7 + t*0.05)
            watcher.complete_step(step, success=True, score=0.9)
    
    print("\n✅ All tests complete!")


# ============================================================================
# ProgressWriter - Writes progress to JSON for web dashboard
# ============================================================================

PROGRESS_FILE = "/tmp/cluster_progress.json"

class ProgressWriter:
    """Writes cluster progress to JSON file for web dashboard and tmux monitor."""
    
    def __init__(self, step_name: str, total_jobs: int = 100, total_seeds: int = 0):
        self.step_name = step_name
        self.total_jobs = total_jobs
        self.total_seeds = total_seeds
        self.jobs_completed = 0
        self.seeds_completed = 0
        self.start_time = time.time()
        self.nodes = {}
        self.finished = False
        self._write()
    
    def register_node(self, hostname: str, gpu_type: str, gpu_count: int):
        """Register a cluster node."""
        self.nodes[hostname] = {
            "total_gpus": gpu_count,
            "gpu_type": gpu_type,
            "jobs_completed": 0,
            "current_seeds_per_sec": 0,
            "last_update": time.time()
        }
        self._write()
    
    def log_gpu_result(self, hostname: str, gpu_id: int, gpu_type: str, seeds_processed: int, duration: float, success: bool = True):
        """Log a completed GPU job result."""
        if hostname in self.nodes:
            self.nodes[hostname]["jobs_completed"] += 1
            if duration > 0:
                self.nodes[hostname]["current_seeds_per_sec"] = seeds_processed / duration
            self.nodes[hostname]["last_update"] = time.time()
        self.seeds_completed += seeds_processed
        self._write()
    
    def update_step(self, step_name: str, total_seeds: int = None):
        """Update the current step name without resetting progress."""
        self.step_name = step_name
        if total_seeds:
            self.total_seeds = total_seeds
            self.seeds_completed = 0
        self.jobs_completed = 0
        self._write()

    def update_progress(self, jobs_done: int = None, chunks_total: int = None, 
                       seeds_done: int = None, message: str = ""):
        """Update overall progress."""
        if jobs_done is not None:
            self.jobs_completed = jobs_done
        if chunks_total is not None:
            self.total_jobs = chunks_total
        if seeds_done is not None:
            self.seeds_completed = seeds_done
        self._write()
    
    def finish(self):
        """Mark the job as finished."""
        self.finished = True
        self._write()
    

    def update_trial_stats(self, trial_num: int = 0, forward_survivors: int = 0, 
                          reverse_survivors: int = 0, bidirectional: int = 0,
                          best_bidirectional: int = 0, config_desc: str = ""):
        """Update current trial statistics."""
        self.trial_stats = {
            "trial_num": trial_num,
            "forward_survivors": forward_survivors,
            "reverse_survivors": reverse_survivors,
            "bidirectional": bidirectional,
            "best_bidirectional": best_bidirectional,
            "config_desc": config_desc
        }
        self._write()

    def _write(self):
        """Write current state to JSON file."""
        import json
        state = {
            "step_name": self.step_name,
            "total_jobs": self.total_jobs,
            "jobs_completed": self.jobs_completed,
            "seeds_completed": self.seeds_completed,
            "total_seeds": self.total_seeds,
            "elapsed_seconds": time.time() - self.start_time,
            "updated_at": time.time(),
            "finished": self.finished,
            "nodes": self.nodes,
            "trial_stats": getattr(self, 'trial_stats', {})
        }
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            pass  # Silently fail if can't write
