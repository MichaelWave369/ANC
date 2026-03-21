from __future__ import annotations

import json
from pathlib import Path

from anchor_sim_v0_2 import SimConfig as V02Config, run_once as run_v02
from anchor_sim_v0_3 import SimConfig as V03Config, run_training
from anc.continuity import (
    branch_stability_score,
    build_diagnostics,
    build_epsilon_signature,
    build_memory_crystal,
    build_state_vector,
)
from anc.tiekat_v57 import ContinuumVault, compare_simulation_paths


def test_v02_still_runs(tmp_path: Path):
    cfg = V02Config(epochs=12, n_validators=10, n_delegators=20, run_id="v02_check", out_dir=str(tmp_path))
    result = run_v02(cfg)
    assert result["summary"]["version"] in {"0.2.0", "0.3.0"}
    assert (tmp_path / "v02_check_metrics.csv").exists()


def test_v57_continuity_primitives_and_branch_scoring():
    rows = [
        {"mean_psi_b": 0.6, "mean_lt": 0.55, "permit": 1},
        {"mean_psi_b": 0.62, "mean_lt": 0.57, "permit": 1},
    ]
    sig = build_epsilon_signature(1, 2, "normal_validation", rows)
    state = build_state_vector(0.61, 0.58, branch_stability_score([0.59, 0.61]), 0.55)
    diag = build_diagnostics(state, "normal_validation")
    crystal = build_memory_crystal("run", 2, sig, state, diag)

    ranked = compare_simulation_paths([("a", diag), ("b", build_diagnostics(state, "recovery_bias", weak_threshold=0.4))])
    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]
    assert crystal.crystal_id.startswith("run:2:")


def test_vault_persistence(tmp_path: Path):
    vault_path = tmp_path / "vault.json"
    vault = ContinuumVault(path=vault_path)
    sig = build_epsilon_signature(1, 2, "evt", [{"mean_psi_b": 0.5, "mean_lt": 0.5, "permit": 1}])
    state = build_state_vector(0.5, 0.5, 0.9, 0.5)
    diag = build_diagnostics(state, "evt")
    vault.append(build_memory_crystal("r", 2, sig, state, diag))
    vault.save()

    loaded = ContinuumVault(path=vault_path)
    loaded.load()
    assert loaded.latest() is not None
    assert loaded.latest().epsilon_signature.regime_event == "evt"


def test_recursive_training_stable_or_improves(tmp_path: Path):
    cfg = V03Config(
        epochs=48,
        continuity_window=12,
        training_runs=3,
        n_validators=16,
        n_delegators=40,
        run_id="v03_train",
        out_dir=str(tmp_path),
    )
    result = run_training(cfg)
    training = result["training"]
    history = training["history"]
    assert len(history) == 3
    baselines = [float(item["baseline"]) for item in history]
    assert baselines == sorted(baselines)

    training_path = Path(result["training_path"])
    assert training_path.exists()
    loaded = json.loads(training_path.read_text(encoding="utf-8"))
    assert loaded["best_continuity_score"] >= 0.0
