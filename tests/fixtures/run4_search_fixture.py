"""Run-4 Step 1 SEARCH fixture — the collaborator stage 1 added to Step 1.

Since `915d382` (Step 1 search record, stage 1) WATCHER schedules Step 1 from the ONE open
search for the family and mode set: it looks the search up, compares the dataset and
every configuration field against the record, assigns `seed_start` = the search's first
gap and `max_seeds` = min(chunk, range end - start), and routes `--step1-search-id`
(agents/watcher_agent.py, run_step, `if step == 1:` search block). A stub database with
no search lookup therefore makes Step 1 fail closed before any command is built, which
is why the Run-4 routing evidence went red.

This module supplies that collaborator for the Run-4 fixtures, through PRODUCTION code
only:
  * the record is written by the production opener, `Step1SearchLedger.open_search`,
    into a THROWAWAY sqlite file under the checkout's data/ directory;
  * its configuration is computed by the production functions WATCHER's own open
    command uses — `_step1_cli_final_params` then `step1_presented_configuration` — so
    run_step's comparison (S4) is the real one and nothing about it is relaxed;
  * the range is Run 4's: [2^31, 2^32); the first gap is 2^31; the chunk is the
    params' max_seeds (2^31). The built command therefore carries
    `--seed-start 2147483648 --max-seeds 2147483648`.

The dataset the search is opened on is the stub frozen dataset the caller's fixture
hands WATCHER (`STUB_DATASET`), so run_step's dataset comparison is also the real one.

The search kind follows the presentation: a pinned build gets a pinned search with the
seven pins; an unpinned build gets an unpinned search over the same range. A build whose
presentation does not match the record refuses before any command exists (stage-1 gate
G5 covers that refusal); the routing gates test what reaches the command, so each build
is given the search its presentation matches.
"""
import os
import shutil
import tempfile
import types

RUN4_RANGE = (2 ** 31, 2 ** 32)

# The frozen-dataset identity the Run-4 fixtures hand WATCHER. A search records four
# identity fields (Step1SearchLedger.open_search refuses any that is empty), so the stub
# carries all four; run_step compares version and sha256.
STUB_DATASET = dict(
    sha256="0" * 64, size=0, record_count=0, manifest_id="run4-fixture",
    version_id="daily3-run4-fixture", manifest_sha256="1" * 64,
    dataset_lineage_id="run4-fixture-lineage")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRATCH = os.path.join(_ROOT, "data", "run4_fixture_tmp")


def stub_frozen_dataset(path):
    return types.SimpleNamespace(path=path, **STUB_DATASET)


class SearchDB:
    """Stands in for `database_system.DistributedPRNGDatabase` in the Run-4 fixtures:
    the stage-1 search lookup over one throwaway ledger, plus the S145 history cursor the
    fixtures already stubbed (empty history: next gap 0)."""

    ledger_path = None          # set by open_run4_search, per build
    pin_bundle = None           # set by the pinned-build harness around one build

    class _Cur:
        is_complete = False
        domain_start = 0
        domain_end_exclusive = 4294967296
        covered_seed_count = 0
        certified_interval_count = 0
        next_seed_start = 0

    def __init__(self, *a, **kw):
        pass

    def get_certified_cursor(self, *a, **kw):
        return self._Cur()

    def step1_search_ledger(self):
        from utils.step1_search_ledger import Step1SearchLedger
        if SearchDB.ledger_path is None:
            raise RuntimeError("run4 fixture: no search was opened for this build")
        return Step1SearchLedger(SearchDB.ledger_path)

    def select_step1_search(self, prng_type, *, test_both_modes):
        from utils.seed_coverage_ledger import canonical_coverage_identity
        base, modes = canonical_coverage_identity(prng_type, test_both_modes=test_both_modes)
        return self.step1_search_ledger().select(base, sorted(modes))


def open_run4_search(agent_mod, agent, params, pin_bundle):
    """Open the build's search in a fresh throwaway ledger and point SearchDB at it.
    Returns a cleanup callable. `params` are the ordinary params run_step receives;
    `pin_bundle` is the frozen operator bundle or None."""
    os.makedirs(_SCRATCH, exist_ok=True)
    d = tempfile.mkdtemp(prefix="run4_search_", dir=_SCRATCH)
    path = os.path.join(d, "prng_analysis.db")
    from utils.step1_search_ledger import Step1SearchLedger
    # The fixture must not change the process it observes. Computing the configuration
    # imports window_optimizer, whose import chain (sklearn -> scipy.stats ->
    # numpy.testing -> numpy.distutils.log) sets the ROOT logger to WARNING. Done here,
    # BEFORE run_step, that would suppress WATCHER's INFO pin-provenance record at
    # agents/watcher_agent.py:2036, which production emits before its own search
    # comparison first imports window_optimizer. The root level is restored. (That
    # production run_step itself loses later INFO records once its comparison imports
    # window_optimizer is a separate finding, reported, not fixed here.)
    import logging
    _root, _level = logging.getLogger(), logging.getLogger().level
    try:
        final = agent_mod._step1_cli_final_params(agent, dict(params), pin_bundle)
        # The pin argument is computed exactly as run_step computes it for its own
        # comparison (`step1_presented_configuration(final_params,
        # _step1_explicit_pin(step, _pin_bundle))`); in production that equals the raw
        # bundle the open command passes.
        presented = agent_mod.step1_presented_configuration(
            final, agent_mod._step1_explicit_pin(1, pin_bundle))
    finally:
        _root.setLevel(_level)
    ds = STUB_DATASET
    Step1SearchLedger(path).open_search(
        configuration=presented, range_start=RUN4_RANGE[0],
        range_end_exclusive=RUN4_RANGE[1],
        dataset={"dataset_version_id": ds["version_id"], "dataset_sha256": ds["sha256"],
                 "input_manifest_sha256": ds["manifest_sha256"],
                 "dataset_lineage_id": ds["dataset_lineage_id"]},
        opened_reason="operator", opened_by_ref="run4 fixture",
        repository_commit="run4-fixture")
    SearchDB.ledger_path = path

    def cleanup():
        SearchDB.ledger_path = None
        shutil.rmtree(d, ignore_errors=True)
    return cleanup
