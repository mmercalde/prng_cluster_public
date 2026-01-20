"""
PATCH: Incremental Output Writing for Window Optimizer
======================================================
Apply this to window_optimizer_bayesian.py

This patch adds a callback that writes best-so-far results after EACH trial,
ensuring crash recovery and WATCHER visibility.

INSTALLATION:
1. Add the create_incremental_save_callback() function to window_optimizer_bayesian.py
2. Modify the study.optimize() call to include the callback
3. Modify the objective function to store survivors in trial.user_attrs

See inline comments for exact insertion points.
"""

# =============================================================================
# PART 1: Add this function to window_optimizer_bayesian.py (near top, after imports)
# =============================================================================

from datetime import datetime
from pathlib import Path
import json

def create_incremental_save_callback(
    output_config_path: str = "optimal_window_config.json",
    output_survivors_path: str = "bidirectional_survivors.json",
    total_trials: int = 50
):
    """
    Factory function that creates an Optuna callback for incremental saving.
    
    The callback writes best-so-far results after each completed trial,
    ensuring crash recovery and WATCHER visibility.
    
    Args:
        output_config_path: Path to write optimal config JSON
        output_survivors_path: Path to write bidirectional survivors JSON  
        total_trials: Total planned trials (for progress reporting)
    
    Returns:
        Callable suitable for study.optimize(callbacks=[...])
    """
    
    def save_best_so_far_callback(study, trial):
        """Invoked after each trial completes."""
        
        completed = len([t for t in study.trials 
                        if t.state.name == "COMPLETE"])
        
        # Build progress metadata
        progress = {
            "status": "in_progress",
            "completed_trials": completed,
            "total_trials": total_trials,
            "last_updated": datetime.now().isoformat(),
            "last_trial_number": trial.number,
            "last_trial_value": trial.value if trial.value is not None else None,
        }
        
        # If we have a best trial, include full config
        if study.best_trial is not None:
            best_config = {
                # Progress tracking
                **progress,
                
                # Best trial info
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "best_bidirectional_count": int(study.best_value) if study.best_value else 0,
                "best_params": study.best_params,
                
                # Extract window parameters for downstream compatibility
                "window_size": study.best_params.get("window_size"),
                "offset": study.best_params.get("offset"),
                "skip_min": study.best_params.get("skip_min"),
                "skip_max": study.best_params.get("skip_max"),
                "forward_threshold": study.best_params.get("forward_threshold"),
                "reverse_threshold": study.best_params.get("reverse_threshold"),
                "time_of_day": study.best_params.get("time_of_day", "all"),
            }
            
            # Write config atomically (write to temp, then rename)
            temp_path = Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(best_config, f, indent=2)
            temp_path.rename(output_config_path)
            
            print(f"📝 Trial {trial.number}: Saved best config (best={study.best_value:.0f} @ trial {study.best_trial.number})")
            
            # If THIS trial is the new best AND has survivors, save them
            if trial.number == study.best_trial.number:
                survivors = trial.user_attrs.get("bidirectional_survivors")
                if survivors is not None and len(survivors) > 0:
                    temp_surv = Path(output_survivors_path).with_suffix(".tmp")
                    with open(temp_surv, "w") as f:
                        json.dump(survivors, f)
                    temp_surv.rename(output_survivors_path)
                    print(f"📝 Trial {trial.number}: Saved {len(survivors)} bidirectional survivors")
        
        else:
            # No successful trial yet - write progress-only file
            progress["best_trial_number"] = None
            progress["best_value"] = None
            progress["note"] = "No successful trials yet"
            
            temp_path = Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(progress, f, indent=2)
            temp_path.rename(output_config_path)
    
    return save_best_so_far_callback


def finalize_output(
    study,
    output_config_path: str = "optimal_window_config.json"
):
    """
    Call this AFTER study.optimize() completes to mark status as 'complete'.
    
    Team Beta requirement: explicit status="complete" finalization.
    """
    if not Path(output_config_path).exists():
        return
    
    with open(output_config_path, "r") as f:
        config = json.load(f)
    
    config["status"] = "complete"
    config["completed_at"] = datetime.now().isoformat()
    
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Finalized {output_config_path} (status=complete)")


# =============================================================================
# PART 2: Modify the objective function to store survivors in trial
# =============================================================================

# Find your objective function (likely named `objective` or similar)
# Add this line BEFORE returning the score:

"""
def objective(trial):
    # ... existing code that runs sieve and gets results ...
    
    bidirectional_survivors = results.get("bidirectional_survivors", [])
    
    # >>> ADD THIS LINE <<<
    trial.set_user_attr("bidirectional_survivors", bidirectional_survivors)
    
    # Return score (survivor count)
    return len(bidirectional_survivors)
"""


# =============================================================================
# PART 3: Modify the study.optimize() call to include callback
# =============================================================================

# Find where study.optimize() is called and change it:

"""
# BEFORE:
study.optimize(objective, n_trials=max_iterations)

# AFTER:
callback = create_incremental_save_callback(
    output_config_path=output_file,  # or "optimal_window_config.json"
    output_survivors_path="bidirectional_survivors.json",
    total_trials=max_iterations
)
study.optimize(objective, n_trials=max_iterations, callbacks=[callback])

# Add finalization after optimize completes:
finalize_output(study, output_config_path=output_file)
"""


# =============================================================================
# PART 4: Quick verification test
# =============================================================================

def test_callback():
    """Quick test to verify callback works."""
    import optuna
    
    # Create dummy study
    study = optuna.create_study(direction="maximize")
    
    # Create callback
    callback = create_incremental_save_callback(
        output_config_path="test_config.json",
        output_survivors_path="test_survivors.json",
        total_trials=5
    )
    
    def dummy_objective(trial):
        x = trial.suggest_int("window_size", 100, 500)
        survivors = [{"seed": i} for i in range(x)]  # Dummy survivors
        trial.set_user_attr("bidirectional_survivors", survivors)
        return len(survivors)
    
    # Run with callback
    study.optimize(dummy_objective, n_trials=5, callbacks=[callback])
    
    # Finalize
    finalize_output(study, "test_config.json")
    
    # Verify
    with open("test_config.json") as f:
        config = json.load(f)
    
    print(f"\n✅ Test passed!")
    print(f"   Status: {config['status']}")
    print(f"   Completed trials: {config['completed_trials']}")
    print(f"   Best value: {config['best_value']}")
    
    # Cleanup
    Path("test_config.json").unlink()
    Path("test_survivors.json").unlink()


if __name__ == "__main__":
    test_callback()
