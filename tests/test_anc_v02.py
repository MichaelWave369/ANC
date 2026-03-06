from __future__ import annotations

import json
from pathlib import Path

from anc.parallax_bridge import ParallaxBridge
from anc.report import ANCReportGenerator
from anc.tiekat_v81 import (
    HEMAVIT_LAMBDA,
    TIEKATVector,
    compute_lt,
    hemavit_path_integral,
    hqrma_flow,
)
from anchor_sim_v0_2 import SimConfig, run_once


def test_hemavit_path_integral_bounded_empty_and_decay_behavior():
    assert hemavit_path_integral([]) == 0.5
    v = hemavit_path_integral([0.2, 0.8])
    assert 0.0 <= v <= 1.0
    # recency weighting: recent drop lowers output
    hi_then_low = hemavit_path_integral([0.95, 0.95, 0.15])
    hi_only = hemavit_path_integral([0.95, 0.95, 0.95])
    assert hi_then_low < hi_only
    assert hemavit_path_integral([0.9, 0.95, 1.0]) > 0.85


def test_hemavit_lambda_constant():
    assert HEMAVIT_LAMBDA == 0.369


def test_hqrma_flow_bounded_and_reasonable_ordering():
    inp = [0.1, 0.3, 0.6, 0.9]
    out = hqrma_flow(inp, eta=0.05, steps=5)
    assert len(out) == len(inp)
    assert all(0.0 <= x <= 1.0 for x in out)
    assert out[0] <= out[1] <= out[2] <= out[3]
    # anti-free-riding: weakest doesn't leapfrog strongest
    assert out[0] < out[-1]


def test_tiekat_vector_shape_entropy_alignment():
    vec = TIEKATVector(
        on_time_ratio=0.95,
        late_ratio=0.02,
        missed_ratio=0.03,
        participation_streak=0.8,
        correctness=0.97,
        equivocation_free=1.0,
        slash_history=0.95,
        uptime_30d=0.96,
        connectivity=0.88,
        decentralization_contrib=0.91,
        delegation_depth=0.72,
        sovereignty_alignment=0.93,
    )
    assert len(vec.as_list()) == 12
    ent = vec.entropy()
    ali = vec.alignment()
    assert 0.0 <= ent <= 1.0
    assert 0.0 <= ali <= 1.0
    assert ali > 0.75


def test_compute_lt_behavior_and_cadence():
    high = compute_lt(1.0, 0.9, 0.05, epoch=9)
    low = compute_lt(1.0, 0.3, 0.5, epoch=1)
    assert high > low
    assert compute_lt(0.0, 1.0, 0.0, epoch=9) == 0.0
    e9 = compute_lt(1.0, 0.8, 0.1, epoch=9)
    e6 = compute_lt(1.0, 0.8, 0.1, epoch=6)
    e3 = compute_lt(1.0, 0.8, 0.1, epoch=3)
    e1 = compute_lt(1.0, 0.8, 0.1, epoch=1)
    assert e9 > e6 > e3 > e1


def test_bridge_schema_and_noop_calls():
    bridge = ParallaxBridge()
    schema = bridge.export_network_lt([], epoch=10, permit=False)
    for key in {
        "network_lt",
        "top_validator",
        "top_lt",
        "bottom_validator",
        "bottom_lt",
        "sovereign_count",
        "total_validators",
        "permit",
        "epoch",
        "tiekat_version",
    }:
        assert key in schema
    bridge.store_epoch_in_tbrc({"epoch": 1})
    bridge.store_simulation_run({"run": "x"})
    bridge.notify_resonance_moment(369)


def test_simulation_smoke_and_determinism(tmp_path: Path):
    cfg = SimConfig(
        epochs=25,
        n_validators=20,
        n_delegators=50,
        run_id="smoke",
        out_dir=str(tmp_path),
        generate_report=True,
        generate_gabriel_summary=True,
    )
    r1 = run_once(cfg)
    r2 = run_once(cfg)
    assert r1["summary"]["final_mean_lt"] == r2["summary"]["final_mean_lt"]
    assert r1["summary"]["final_mean_psi_b"] == r2["summary"]["final_mean_psi_b"]

    for suffix in ["_metrics.csv", "_summary.json", "_bridge.json", "_report.md", "_gabriel.md", "_lt_chart.txt"]:
        assert (tmp_path / f"smoke{suffix}").exists()


def test_cartel_sybil_and_resonance(tmp_path: Path):
    cfg = SimConfig(
        epochs=369,
        n_validators=30,
        n_delegators=60,
        run_id="attack_modes",
        out_dir=str(tmp_path),
        enable_cartel=True,
        enable_sybil=True,
        generate_report=False,
        generate_gabriel_summary=False,
    )
    result = run_once(cfg)
    summary = result["summary"]
    assert summary["cartel_enabled"] is True
    assert summary["sybil_enabled"] is True
    assert 369 in summary["resonance_events"]


def test_report_generation_direct(tmp_path: Path):
    gen = ANCReportGenerator()
    summary = {
        "run_id": "r",
        "epochs": 2,
        "n_validators": 3,
        "n_delegators": 4,
        "seed": 369369,
        "final_mean_lt": 0.5,
        "final_mean_psi_b": 0.6,
        "permit_rate": 1.0,
        "resonance_events": [],
        "use_hemavit_issuance": True,
        "use_hqrma_smoothing": True,
        "use_12axis_telemetry": True,
    }
    rows = [{"mean_lt": 0.4, "mean_psi_b": 0.5}, {"mean_lt": 0.5, "mean_psi_b": 0.6}]
    rp = gen.generate_report(summary, rows, output_dir=str(tmp_path))
    gp = gen.generate_gabriel_summary(summary, rows, output_dir=str(tmp_path))
    bp = gen.generate_bridge_export({"x": 1}, output_dir=str(tmp_path), run_id="r")
    assert Path(rp).exists()
    assert Path(gp).exists()
    assert Path(bp).exists()
    loaded = json.loads(Path(bp).read_text(encoding="utf-8"))
    assert loaded["x"] == 1
