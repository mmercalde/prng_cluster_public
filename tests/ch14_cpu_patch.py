"""CPU-forcing patch for the CH14-DIAG wave 1 tests. Imports no code under test."""
import contextlib
import os

TREE_PARAMS = {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.1}


@contextlib.contextmanager
def force_cpu_trees():
    """The trial worker asks for GPU devices; force CPU for the tests (same patch on and off)."""
    import xgboost, lightgbm, catboost
    ox, ol, oc = xgboost.train, lightgbm.train, catboost.CatBoostRegressor

    def x_train(params, *a, **k):
        q = dict(params); q["device"] = "cpu"
        return ox(q, *a, **k)

    def l_train(params, *a, **k):
        q = dict(params); q["device"] = "cpu"; q.pop("gpu_platform_id", None); q.pop("gpu_device_id", None)
        q["num_threads"] = len(os.sched_getaffinity(0))  # cores this process may use (taskset)
        return ol(q, *a, **k)

    class CPUCat(oc):
        def __init__(self, *a, **k):
            k["task_type"] = "CPU"; k.pop("devices", None)
            super().__init__(*a, **k)

    xgboost.train, lightgbm.train, catboost.CatBoostRegressor = x_train, l_train, CPUCat
    try:
        yield
    finally:
        xgboost.train, lightgbm.train, catboost.CatBoostRegressor = ox, ol, oc
