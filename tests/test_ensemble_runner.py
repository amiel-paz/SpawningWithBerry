import argparse
import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "examples/ethylene_pyscf/run_ensemble.py"
    spec = importlib.util.spec_from_file_location("ethylene_run_ensemble", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(tmp_path):
    (tmp_path / "h.xyz").write_text("1\nensemble supervisor\nH 0 0 0\n")
    input_path = tmp_path / "production.in"
    input_path.write_text(
        "provider custom\ngeometry h.xyz\nnum_states 1\ninitial_state 0\n"
        "time_step 1 au\nsimulation_time 1 au\n"
        "electronic_method custom\nrun_directory run-production\n"
    )
    return argparse.Namespace(
        input=input_path,
        ensemble_name="run-ensemble",
        simulation_time=None,
        time_step=None,
        pair_overlap_threshold=None,
        memory_limit_gb=32.0,
        runtime_limit_hours=72.0,
        production_count=2,
        parallel_members=2,
    )


def test_member_failure_is_quarantined_without_stopping_siblings(tmp_path, monkeypatch):
    module = _module()
    args = _args(tmp_path)
    exit_codes = {1: 1, 2: 0}

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            self.seed = int(command[command.index("--member-seed") + 1])
            self.pid = 1000 + self.seed

        def poll(self):
            return exit_codes[self.seed]

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(module, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    durations, failures = module._run_batch(args, [1, 2], workers=1)
    assert len(durations) == 1
    assert [item["seed"] for item in failures] == [1]
    status = json.loads(
        (tmp_path / "run-ensemble/ensemble-status.json").read_text()
    )
    assert status["active_seeds"] == []
    assert status["completed_seeds"] == [2]
    assert [item["seed"] for item in status["failed_members"]] == [1]
    assert (tmp_path / "run-ensemble/seed-1/failure.json").is_file()


def test_existing_failure_marker_skips_only_that_seed(tmp_path, monkeypatch):
    module = _module()
    args = _args(tmp_path)
    failed = tmp_path / "run-ensemble/seed-1"
    failed.mkdir(parents=True)
    (failed / "failure.json").write_text(
        json.dumps({"seed": 1, "exit_code": 1, "progress": {"time_au": 0.5}})
    )
    launched = []

    class FakeProcess:
        pid = 1002

        def __init__(self, command, **_kwargs):
            launched.append(int(command[command.index("--member-seed") + 1]))

        def poll(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(module, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    _durations, failures = module._run_batch(args, [1, 2], workers=1)
    assert launched == [2]
    assert [item["seed"] for item in failures] == [1]
