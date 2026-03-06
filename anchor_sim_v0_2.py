#!/usr/bin/env python3
"""Anchor (ANC) simulation v0.2.0 (TIEKAT v8.1).

Side-by-side upgrade path for v0.1.1a. This file intentionally does not modify
`anchor_sim_v0_1_1a.py` and preserves fallback knobs for legacy-like behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from anc import __seed__, __tiekat_version__, __version__
from anc.parallax_bridge import ParallaxBridge
from anc.report import ANCReportGenerator
from anc.tiekat_v81 import (
    HEMAVIT_LAMBDA,
    TIEKATVector,
    compute_lt,
    hemavit_path_integral,
    hqrma_flow,
)


@dataclass
class SimConfig:
    epochs: int = 5000
    seed: int = __seed__
    run_id: str = "anchor_sim_v0_2"

    n_validators: int = 120
    n_delegators: int = 4000
    initial_total_stake: float = 100_000_000.0

    validator_stake_lognorm_mu: float = 0.0
    validator_stake_lognorm_sigma: float = 1.0
    delegator_stake_lognorm_mu: float = -0.5
    delegator_stake_lognorm_sigma: float = 1.2

    # v0.2 fields
    tiekat_version: str = __tiekat_version__
    hemavit_lambda: float = HEMAVIT_LAMBDA
    hqrma_eta: float = 0.05
    hqrma_steps: int = 3
    psi_history_depth: int = 369
    use_12axis_telemetry: bool = True
    use_hqrma_smoothing: bool = True
    use_hemavit_issuance: bool = True
    track_lt_per_validator: bool = True
    generate_report: bool = True
    generate_gabriel_summary: bool = True

    # legacy-compatible behavior
    ema_alpha: float = 0.2
    k_psi: float = 0.1
    psi_emit: float = 0.45
    alpha_min: float = 0.55

    enable_shocks: bool = True
    shock_every_epochs: int = 333
    shock_fraction_validators: float = 0.18
    shock_duration_epochs: int = 3

    enable_cartel: bool = False
    cartel_fraction: float = 0.12
    enable_sybil: bool = False
    sybil_count: int = 60
    sybil_stake_each: float = 25_000.0

    out_dir: str = "out"


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def gini(values: List[float]) -> float:
    vals = sorted([max(0.0, v) for v in values])
    if not vals:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    n = len(vals)
    return (2.0 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * total)) - (n + 1) / n


def nakamoto_coefficient(weights: List[float], threshold: float) -> int:
    ws = sorted((max(0.0, w) for w in weights), reverse=True)
    total = sum(ws)
    if total <= 0:
        return 0
    target = threshold * total
    acc = 0.0
    for i, w in enumerate(ws, start=1):
        acc += w
        if acc >= target:
            return i
    return len(ws)


def topk_share(values: List[float], k: int) -> float:
    vals = sorted((max(0.0, v) for v in values), reverse=True)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return sum(vals[: max(0, k)]) / total


def effective_stake(stake: float, psi_b: float, capture_c: float, k_psi: float) -> float:
    return max(0.0, stake * (1.0 + k_psi * clamp01(psi_b)) * (1.0 - clamp01(capture_c)))


@dataclass
class Validator:
    vid: int
    stake_bonded: float
    stake_delegated: float
    opq: float
    capture_c: float = 0.0
    psi_b: float = 0.55
    alpha: float = 0.7
    permit: bool = True

    tiekat_vec: Optional[TIEKATVector] = None
    lt_score: float = 0.0
    lt_history: List[float] = field(default_factory=list)
    psi_b_history: List[float] = field(default_factory=list)
    participation_streak: int = 0

    cartel_member: bool = False
    sybil_member: bool = False


@dataclass
class Delegator:
    did: int
    stake: float
    validator_id: int


def _sample_lognormal(rng: random.Random, mu: float, sigma: float, n: int) -> List[float]:
    return [rng.lognormvariate(mu, sigma) for _ in range(n)]


def init_validators(cfg: SimConfig, rng: random.Random) -> List[Validator]:
    raw = _sample_lognormal(rng, cfg.validator_stake_lognorm_mu, cfg.validator_stake_lognorm_sigma, cfg.n_validators)
    scale = cfg.initial_total_stake * 0.6 / max(1e-9, sum(raw))
    vals = [
        Validator(
            vid=i,
            stake_bonded=raw[i] * scale,
            stake_delegated=0.0,
            opq=clamp01(rng.gauss(0.85, 0.08)),
        )
        for i in range(cfg.n_validators)
    ]

    if cfg.enable_cartel:
        cartel_n = max(1, int(cfg.cartel_fraction * cfg.n_validators))
        for v in vals[:cartel_n]:
            v.cartel_member = True
            v.capture_c = 0.55

    if cfg.enable_sybil:
        for i in range(min(cfg.sybil_count, len(vals))):
            vals[-(i + 1)].sybil_member = True
            vals[-(i + 1)].stake_bonded = cfg.sybil_stake_each
            vals[-(i + 1)].capture_c = 0.75

    return vals


def init_delegators(cfg: SimConfig, rng: random.Random, validators: List[Validator]) -> List[Delegator]:
    raw = _sample_lognormal(rng, cfg.delegator_stake_lognorm_mu, cfg.delegator_stake_lognorm_sigma, cfg.n_delegators)
    scale = cfg.initial_total_stake * 0.4 / max(1e-9, sum(raw))
    delegators: List[Delegator] = []
    for i, rs in enumerate(raw):
        vid = rng.randrange(len(validators))
        stake = rs * scale
        validators[vid].stake_delegated += stake
        delegators.append(Delegator(did=i, stake=stake, validator_id=vid))
    return delegators


def maybe_apply_shock(cfg: SimConfig, epoch: int, rng: random.Random, validators: List[Validator]) -> set[int]:
    if not cfg.enable_shocks or epoch == 0 or epoch % cfg.shock_every_epochs != 0:
        return set()
    n = max(1, int(cfg.shock_fraction_validators * len(validators)))
    return set(rng.sample([v.vid for v in validators], k=n))


def is_shocked(shocked_until: Dict[int, int], vid: int, epoch: int) -> bool:
    return shocked_until.get(vid, -1) >= epoch


def _build_tiekat_vector(v: Validator, shocked: bool, rng: random.Random) -> TIEKATVector:
    miss = 0.22 if shocked else clip(0.03 + rng.random() * 0.08, 0.0, 1.0)
    late = clip(0.05 + rng.random() * 0.08 + (0.12 if shocked else 0.0), 0.0, 1.0)
    on_time = clamp01(1.0 - late - miss)
    correctness = clamp01(v.opq - (0.12 if shocked else 0.01 * rng.random()))
    equiv_free = clamp01(0.98 - (0.2 if v.sybil_member else 0.0))
    slash_hist = clamp01(1.0 - (0.35 if (v.cartel_member or v.sybil_member) else 0.05))
    uptime = clamp01(v.opq - (0.15 if shocked else 0.0))
    connectivity = clamp01(0.75 + 0.2 * rng.random())
    decentralization = clamp01(0.3 if v.cartel_member else 0.9 if not v.sybil_member else 0.2)
    delegation_depth = clamp01(1.0 - min(1.0, v.stake_delegated / 5_000_000.0))
    sovereignty = clamp01(0.25 if (v.cartel_member or v.sybil_member) else 0.9)
    streak = clamp01(v.participation_streak / 100.0)
    return TIEKATVector(
        on_time_ratio=on_time,
        late_ratio=late,
        missed_ratio=miss,
        participation_streak=streak,
        correctness=correctness,
        equivocation_free=equiv_free,
        slash_history=slash_hist,
        uptime_30d=uptime,
        connectivity=connectivity,
        decentralization_contrib=decentralization,
        delegation_depth=delegation_depth,
        sovereignty_alignment=sovereignty,
    )


def simulate_epoch(
    cfg: SimConfig,
    epoch: int,
    rng: random.Random,
    validators: List[Validator],
    shocked_until: Dict[int, int],
) -> Tuple[Dict[str, float], bool]:
    shocked_now = maybe_apply_shock(cfg, epoch, rng, validators)
    for vid in shocked_now:
        shocked_until[vid] = epoch + cfg.shock_duration_epochs - 1

    psi_raw = []
    alpha_raw = []
    permits = []

    for v in validators:
        shocked = is_shocked(shocked_until, v.vid, epoch)
        if cfg.use_12axis_telemetry:
            v.tiekat_vec = _build_tiekat_vector(v, shocked, rng)
            entropy = v.tiekat_vec.entropy()
            alpha = v.tiekat_vec.alignment()
            psi = clamp01((alpha * (1.0 - 0.5 * entropy) * (0.8 + 0.2 * v.opq)))
        else:
            # legacy style fallback (simple 5-axis proxy)
            alpha = clamp01(0.6 * v.opq + 0.4 * (1.0 - (0.3 if shocked else 0.05)))
            psi = clamp01((0.5 * alpha + 0.5 * v.psi_b) * (1.0 - 0.2 * v.capture_c))

        v.participation_streak = v.participation_streak + 1 if not shocked else 0
        psi_raw.append(psi)
        alpha_raw.append(alpha)

    if cfg.use_hqrma_smoothing:
        psi_smooth = hqrma_flow(psi_raw, eta=cfg.hqrma_eta, steps=cfg.hqrma_steps)
    else:
        psi_smooth = [clamp01(cfg.ema_alpha * p + (1.0 - cfg.ema_alpha) * validators[i].psi_b) for i, p in enumerate(psi_raw)]

    lt_vals = []
    eff_stakes = []
    for i, v in enumerate(validators):
        v.alpha = alpha_raw[i]
        v.psi_b = psi_smooth[i]
        v.psi_b_history.append(v.psi_b)
        if len(v.psi_b_history) > cfg.psi_history_depth:
            v.psi_b_history.pop(0)

        gate_psi = hemavit_path_integral(v.psi_b_history, lambda_decay=cfg.hemavit_lambda) if cfg.use_hemavit_issuance else v.psi_b
        v.permit = gate_psi >= cfg.psi_emit and v.alpha >= cfg.alpha_min
        permits.append(v.permit)

        a_on = 0.0 if (v.cartel_member and v.capture_c > 0.7) else 1.0
        if cfg.track_lt_per_validator:
            v.lt_score = compute_lt(a_on, v.psi_b, v.capture_c, epoch)
            v.lt_history.append(v.lt_score)
        lt_vals.append(v.lt_score)
        eff_stakes.append(effective_stake(v.stake_bonded + v.stake_delegated, v.psi_b, v.capture_c, cfg.k_psi))

    permit = (sum(1 for p in permits if p) / len(permits)) >= 0.5 if permits else False
    metrics = {
        "epoch": epoch,
        "mean_psi_b": sum(v.psi_b for v in validators) / len(validators),
        "mean_lt": sum(lt_vals) / len(lt_vals) if lt_vals else 0.0,
        "permit": int(permit),
        "nakamoto_33": nakamoto_coefficient(eff_stakes, 0.33),
        "gini_stake": gini(eff_stakes),
        "top10_share": topk_share(eff_stakes, 10),
    }
    return metrics, permit


def run_once(cfg: SimConfig) -> Dict[str, object]:
    rng = random.Random(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validators = init_validators(cfg, rng)
    delegators = init_delegators(cfg, rng, validators)
    bridge = ParallaxBridge()
    reporter = ANCReportGenerator()

    shocked_until: Dict[int, int] = {}
    metrics_rows: List[Dict[str, float]] = []
    resonance_events: List[int] = []

    for epoch in range(1, cfg.epochs + 1):
        row, permit = simulate_epoch(cfg, epoch, rng, validators, shocked_until)
        metrics_rows.append(row)
        if epoch % 369 == 0:
            resonance_events.append(epoch)
            bridge.notify_resonance_moment(epoch)

    network_lt = bridge.export_network_lt(validators, cfg.epochs, permit=bool(metrics_rows[-1]["permit"]))
    bridge.store_epoch_in_tbrc(network_lt)

    summary = {
        "run_id": cfg.run_id,
        "version": __version__,
        "tiekat_version": cfg.tiekat_version,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "n_validators": len(validators),
        "n_delegators": len(delegators),
        "final_mean_lt": metrics_rows[-1]["mean_lt"] if metrics_rows else 0.0,
        "final_mean_psi_b": metrics_rows[-1]["mean_psi_b"] if metrics_rows else 0.0,
        "permit_rate": (sum(r["permit"] for r in metrics_rows) / len(metrics_rows)) if metrics_rows else 0.0,
        "nakamoto_33": metrics_rows[-1]["nakamoto_33"] if metrics_rows else 0,
        "gini_stake": metrics_rows[-1]["gini_stake"] if metrics_rows else 0.0,
        "resonance_events": resonance_events,
        "use_12axis_telemetry": cfg.use_12axis_telemetry,
        "use_hqrma_smoothing": cfg.use_hqrma_smoothing,
        "use_hemavit_issuance": cfg.use_hemavit_issuance,
        "bridge_status": bridge.status,
        "cartel_enabled": cfg.enable_cartel,
        "sybil_enabled": cfg.enable_sybil,
    }

    metrics_path = out_dir / f"{cfg.run_id}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()) if metrics_rows else ["epoch"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary_path = out_dir / f"{cfg.run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    bridge_path = reporter.generate_bridge_export(network_lt, output_dir=cfg.out_dir, run_id=cfg.run_id)
    report_path = ""
    gabriel_path = ""
    if cfg.generate_report:
        report_path = reporter.generate_report(summary, metrics_rows, output_dir=cfg.out_dir)
    if cfg.generate_gabriel_summary:
        gabriel_path = reporter.generate_gabriel_summary(summary, metrics_rows, output_dir=cfg.out_dir)

    bridge.store_simulation_run({"summary": summary, "bridge": network_lt})

    print("\n=== ANC v0.2 simulation complete ===")
    print(f"run_id: {cfg.run_id}")
    print(f"epochs: {cfg.epochs}")
    print(f"final mean L(t): {summary['final_mean_lt']:.6f}")
    print(f"final mean Psi_b: {summary['final_mean_psi_b']:.6f}")
    print(f"permit rate: {summary['permit_rate']:.6f}")
    print(f"metrics: {metrics_path}")
    print(f"summary: {summary_path}")
    print(f"bridge: {bridge_path}")
    if report_path:
        print(f"report: {report_path}")
    if gabriel_path:
        print(f"gabriel: {gabriel_path}")

    return {
        "summary": summary,
        "metrics_rows": metrics_rows,
        "paths": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "bridge": bridge_path,
            "report": report_path,
            "gabriel": gabriel_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ANC v0.2 simulation")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--run-id", type=str, default="anchor_sim_v0_2")
    parser.add_argument("--seed", type=int, default=__seed__)
    parser.add_argument("--validators", type=int, default=120)
    parser.add_argument("--delegators", type=int, default=4000)
    parser.add_argument("--cartel", action="store_true")
    parser.add_argument("--sybil", action="store_true")
    args = parser.parse_args()

    cfg = SimConfig(
        epochs=args.epochs,
        run_id=args.run_id,
        seed=args.seed,
        n_validators=args.validators,
        n_delegators=args.delegators,
        enable_cartel=args.cartel,
        enable_sybil=args.sybil,
    )
    run_once(cfg)


if __name__ == "__main__":
    main()
