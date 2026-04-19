"""
fakes.py
Fake MultiGPUCoordinator and support objects for n_parallel structural harness.
Captures all constructor args and runtime attribute assignments.
"""
import threading

class FakeSSHPool:
    def cleanup_all(self): pass
    def close(self): pass

class FakeCoordinator:
    """
    Captures all attributes assigned after construction.
    Fulfills the interface expected by window_optimizer_integration_final.py
    partition worker path.
    """
    _instances = []
    _lock = threading.Lock()

    def __init__(self, config_file=None, **kwargs):
        self.config_file = config_file
        self._init_kwargs = kwargs
        self.ssh_pool = FakeSSHPool()
        self.results = []
        self.node_allowlist = kwargs.get('node_allowlist', [])
        # Seed caps — may be passed via constructor or set after construction
        self.seed_cap_amd    = kwargs.get('seed_cap_amd', None)
        self.seed_cap_nvidia = kwargs.get('seed_cap_nvidia', None)
        # Runtime flags — set by integration layer after construction
        self.use_persistent_workers = kwargs.get('use_persistent_workers', None)
        self.use_zmq_sqlite         = kwargs.get('use_zmq_sqlite', None)
        self.pwc_transport          = kwargs.get('pwc_transport', None)
        self.pwc_min_workers        = kwargs.get('pwc_min_workers', None)
        self.worker_pool_size       = kwargs.get('worker_pool_size', None)
        self.prng_type   = kwargs.get('prng_type', None)
        self.lottery_file = kwargs.get('lottery_file', None)
        self.methods_called = {
            'load_configuration': 0,
            'create_gpu_workers': 0,
            'execute_distributed_analysis': 0,
            'optimize_window': 0,
        }
        with FakeCoordinator._lock:
            FakeCoordinator._instances.append(self)

    def load_configuration(self):
        self.methods_called['load_configuration'] += 1

    def create_gpu_workers(self):
        self.methods_called['create_gpu_workers'] += 1

    def execute_distributed_analysis(self, *args, **kwargs):
        self.methods_called['execute_distributed_analysis'] += 1
        return {'survivors': [], 'forward': [], 'reverse': []}

    def snapshot(self):
        """Return a dict of all captured runtime state."""
        return {
            'config_file':            self.config_file,
            'node_allowlist':         list(self.node_allowlist or []),
            'seed_cap_nvidia':        self.seed_cap_nvidia,
            'seed_cap_amd':           self.seed_cap_amd,
            'use_persistent_workers': self.use_persistent_workers,
            'use_zmq_sqlite':         self.use_zmq_sqlite,
            'pwc_transport':          self.pwc_transport,
            'pwc_min_workers':        self.pwc_min_workers,
            'worker_pool_size':       self.worker_pool_size,
            'methods_called':         dict(self.methods_called),
        }

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instances.clear()

    @classmethod
    def get_instances(cls):
        with cls._lock:
            return list(cls._instances)


class FakeTestResult:
    """Minimal TestResult returned by stubbed run_bidirectional_test."""
    def __init__(self, trial_num=0):
        self.forward_survivors = []
        self.reverse_survivors = []
        self.bidirectional_survivors = []
        self.forward_match_rate = 0.5
        self.reverse_match_rate = 0.5
        self.score = 0.5
        self.trial_number = trial_num
        self.window_size = 6
        self.offset = 54
        self.skip_min = 8
        self.skip_max = 116
        self.skip_mode = 'constant'
        self.prng_type = 'java_lcg'
        self.forward_count = 0.0
        self.reverse_count = 0.0
        self.bidirectional_count = 0.0
        self.intersection_count = 0.0
        self.intersection_ratio = 0.0
        self.intersection_weight = 0.0
        self.bidirectional_selectivity = 0.0
        self.forward_only_count = 0.0
        self.reverse_only_count = 0.0
        self.survivor_overlap_ratio = 0.0
