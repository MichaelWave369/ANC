#!/usr/bin/env python3
"""Anchor (ANC) simulation v0.3.0 powered by TIEKAT v57.7 continuity."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from anc import __seed__, __tiekat_version__, __version__
from anc.continuity import (
    branch_stability_score,
    build_diagnostics,
    build_epsilon_signature,
    build_memory_crystal,
    build_state_vector,
    network_continuity_score,
    recursive_attractor_improvement,
    validator_continuity_score,
)
from anc.parallax_bridge import ParallaxBridge
from anc.report import ANCReportGenerator
from anc.tiekat_v57 import ContinuumVault, compare_simulation_paths, clamp
from anc.tiekat_v81 import HEMAVIT_LAMBDA, TIEKATVector, compute_lt, hemavit_path_integral, hqrma_flow


@dataclass
class SimConfig:
    epochs: int = 5000
    seed: int = __seed__
    run_id: str = "anchor_sim_v0_3"

    n_validators: int = 120
    n_delegators: int = 4000
    initial_total_stake: float = 100_000_000.0

    validator_stake_lognorm_mu: float = 0.0
    validator_stake_lognorm_sigma: float = 1.0
    delegator_stake_lognorm_mu: float = -0.5
    delegator_stake_lognorm_sigma: float = 1.2

    tiekat_version: str = __tiekat_version__
    hemavit_lambda: float = HEMAVIT_LAMBDA
    hqrma_eta: float = 0.05
    hqrma_steps: int = 3
    psi_history_depth: int = 369
    continuity_window: int = 36
    weak_continuity_threshold: float = 0.58
    training_runs: int = 1

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
    generate_report: bool = True
    generate_gabriel_summary: bool = True


def gini(values: List[float]) -> float:
    vals = sorted(max(0.0, v) for v in values)
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
    return (sum(vals[: max(0, k)]) / total) if total > 0 else 0.0


def effective_stake(stake: float, psi_b: float, capture_c: float, k_psi: float) -> float:
    return max(0.0, stake * (1.0 + k_psi * clamp(psi_b)) * (1.0 - clamp(capture_c)))


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
    alpha_history: List[float] = field(default_factory=list)
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
        Validator(vid=i, stake_bonded=raw[i] * scale, stake_delegated=0.0, opq=clamp(rng.gauss(0.85, 0.08)))
        for i in range(cfg.n_validators)
    ]
    if cfg.enable_cartel:
        for v in vals[: max(1, int(cfg.cartel_fraction * cfg.n_validators))]:
            v.cartel_member = True
            v.capture_c = 0.55
    if cfg.enable_sybil:
        for i in range(min(cfg.sybil_count, len(vals))):
            v = vals[-(i + 1)]
            v.sybil_member = True
            v.stake_bonded = cfg.sybil_stake_each
            v.capture_c = 0.75
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


def _build_tiekat_vector(v: Validator, shocked: bool, rng: random.Random, recovery_mode: bool) -> TIEKATVector:
    recovery_bonus = 0.08 if recovery_mode else 0.0
    miss = 0.22 if shocked else clamp(0.03 + rng.random() * 0.08 - recovery_bonus)
    late = clamp(0.05 + rng.random() * 0.08 + (0.12 if shocked else 0.0) - recovery_bonus)
    on_time = clamp(1.0 - late - miss)
    correctness = clamp(v.opq - (0.12 if shocked else 0.01 * rng.random()) + recovery_bonus)
    equiv_free = clamp(0.98 - (0.2 if v.sybil_member else 0.0))
    slash_hist = clamp(1.0 - (0.35 if (v.cartel_member or v.sybil_member) else 0.05))
    uptime = clamp(v.opq - (0.15 if shocked else 0.0) + recovery_bonus)
    connectivity = clamp(0.75 + 0.2 * rng.random() + 0.5 * recovery_bonus)
    decentralization = clamp(0.3 if v.cartel_member else 0.9 if not v.sybil_member else 0.2)
    delegation_depth = clamp(1.0 - min(1.0, v.stake_delegated / 5_000_000.0))
    sovereignty = clamp(0.25 if (v.cartel_member or v.sybil_member) else 0.9)
    streak = clamp(v.participation_streak / 100.0)
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
    recovery_mode: bool,
) -> Tuple[Dict[str, float], bool]:
    for vid in maybe_apply_shock(cfg, epoch, rng, validators):
        shocked_until[vid] = epoch + cfg.shock_duration_epochs - 1

    psi_raw: List[float] = []
    alpha_raw: List[float] = []
    permits: List[bool] = []

    for v in validators:
        shocked = shocked_until.get(v.vid, -1) >= epoch
        v.tiekat_vec = _build_tiekat_vector(v, shocked, rng, recovery_mode)
        entropy = v.tiekat_vec.entropy()
        alpha = v.tiekat_vec.alignment()
        psi = clamp(alpha * (1.0 - 0.5 * entropy) * (0.8 + 0.2 * v.opq))

        v.participation_streak = v.participation_streak + 1 if not shocked else 0
        psi_raw.append(psi)
        alpha_raw.append(alpha)

    psi_smooth = hqrma_flow(psi_raw, eta=cfg.hqrma_eta, steps=cfg.hqrma_steps)

    lt_vals: List[float] = []
    eff_stakes: List[float] = []
    for i, v in enumerate(validators):
        v.alpha = alpha_raw[i]
        v.alpha_history.append(v.alpha)
        v.psi_b = psi_smooth[i]
        v.psi_b_history.append(v.psi_b)
        if len(v.psi_b_history) > cfg.psi_history_depth:
            v.psi_b_history.pop(0)
        gate_psi = hemavit_path_integral(v.psi_b_history, lambda_decay=cfg.hemavit_lambda)
        v.permit = gate_psi >= cfg.psi_emit and v.alpha >= cfg.alpha_min
        permits.append(v.permit)

        a_on = 0.0 if (v.cartel_member and v.capture_c > 0.7) else 1.0
        v.lt_score = compute_lt(a_on, v.psi_b, v.capture_c, epoch)
        v.lt_history.append(v.lt_score)
        lt_vals.append(v.lt_score)
        eff_stakes.append(effective_stake(v.stake_bonded + v.stake_delegated, v.psi_b, v.capture_c, cfg.k_psi))

    permit = (sum(1 for p in permits if p) / len(permits)) >= 0.5 if permits else False
    return {
        "epoch": epoch,
        "mean_psi_b": sum(v.psi_b for v in validators) / len(validators),
        "mean_alpha": sum(v.alpha for v in validators) / len(validators),
        "mean_lt": sum(lt_vals) / len(lt_vals) if lt_vals else 0.0,
        "permit": int(permit),
        "nakamoto_33": nakamoto_coefficient(eff_stakes, 0.33),
        "gini_stake": gini(eff_stakes),
        "top10_share": topk_share(eff_stakes, 10),
    }, permit


def _window(rows: List[Dict[str, float]], size: int) -> List[Dict[str, float]]:
    return rows[-size:] if len(rows) >= size else rows[:]


def run_once(cfg: SimConfig, run_index: int = 0, baseline_score: float = 0.0) -> Dict[str, object]:
    run_seed = cfg.seed + run_index
    run_id = cfg.run_id if run_index == 0 else f"{cfg.run_id}_train{run_index}"
    rng = random.Random(run_seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validators = init_validators(cfg, rng)
    delegators = init_delegators(cfg, rng, validators)
    bridge = ParallaxBridge()
    reporter = ANCReportGenerator()
    vault = ContinuumVault(path=out_dir / f"{run_id}_memory_crystals.json")

    shocked_until: Dict[int, int] = {}
    metrics_rows: List[Dict[str, float]] = []
    continuity_rows: List[Dict[str, float]] = []
    branch_rows: List[Dict[str, float | str]] = []
    resonance_events: List[int] = []
    recovery_mode = False
    recovery_events = 0

    prev_continuity = baseline_score
    for epoch in range(1, cfg.epochs + 1):
        row, permit = simulate_epoch(cfg, epoch, rng, validators, shocked_until, recovery_mode)
        metrics_rows.append(row)
        if epoch % 369 == 0:
            resonance_events.append(epoch)
            bridge.notify_resonance_moment(epoch)

        if epoch % cfg.continuity_window == 0:
            wnd = _window(metrics_rows, cfg.continuity_window)
            regime_event = "shock_recovery" if recovery_mode else "normal_validation"
            sig = build_epsilon_signature(epoch - len(wnd) + 1, epoch, regime_event, wnd)
            val_score = validator_continuity_score((r["mean_psi_b"] for r in wnd), (r["mean_alpha"] for r in wnd))
            net_score = network_continuity_score((r["mean_lt"] for r in wnd), (int(r["permit"]) for r in wnd))

            base_path = clamp(0.5 * val_score + 0.5 * net_score)
            recovery_path = clamp(base_path + (0.06 if recovery_mode else 0.02))
            stability = branch_stability_score([base_path, recovery_path])
            attractor = recursive_attractor_improvement(prev_continuity, base_path)

            state = build_state_vector(val_score, net_score, stability, attractor)
            diag = build_diagnostics(state, regime_event=regime_event, weak_threshold=cfg.weak_continuity_threshold)
            crystal = build_memory_crystal(run_id=run_id, epoch_index=epoch, signature=sig, state=state, diagnostics=diag)
            vault.append(crystal)

            ranked = compare_simulation_paths(
                [("candidate_base", diag), ("candidate_recovery", build_diagnostics(state, "recovery_bias", cfg.weak_continuity_threshold - 0.05))]
            )
            selected_path, selected_score = ranked[0]
            branch_rows.append({"epoch": epoch, "selected_path": selected_path, "selected_score": selected_score})

            continuity_rows.append({
                "epoch": epoch,
                "validator_continuity": val_score,
                "network_continuity": net_score,
                "branch_stability": stability,
                "attractor_gain": attractor,
                "continuity_score": diag.continuity_score,
                "weak_continuity": int(diag.weak_continuity),
            })
            prev_continuity = diag.continuity_score
            recovery_mode = diag.weak_continuity
            if recovery_mode:
                recovery_events += 1

    vault.save()

    network_lt = bridge.export_network_lt(validators, cfg.epochs, permit=bool(metrics_rows[-1]["permit"]))
    bridge.store_epoch_in_tbrc(network_lt)

    training_score = continuity_rows[-1]["continuity_score"] if continuity_rows else 0.0
    summary = {
        "run_id": run_id,
        "version": __version__,
        "tiekat_version": cfg.tiekat_version,
        "seed": run_seed,
        "epochs": cfg.epochs,
        "n_validators": len(validators),
        "n_delegators": len(delegators),
        "final_mean_lt": metrics_rows[-1]["mean_lt"] if metrics_rows else 0.0,
        "final_mean_psi_b": metrics_rows[-1]["mean_psi_b"] if metrics_rows else 0.0,
        "permit_rate": (sum(r["permit"] for r in metrics_rows) / len(metrics_rows)) if metrics_rows else 0.0,
        "nakamoto_33": metrics_rows[-1]["nakamoto_33"] if metrics_rows else 0,
        "gini_stake": metrics_rows[-1]["gini_stake"] if metrics_rows else 0.0,
        "resonance_events": resonance_events,
        "bridge_status": bridge.status,
        "cartel_enabled": cfg.enable_cartel,
        "sybil_enabled": cfg.enable_sybil,
        "continuity_windows": len(continuity_rows),
        "continuity_score": training_score,
        "recovery_events": recovery_events,
        "training_baseline_score": baseline_score,
        "training_delta": training_score - baseline_score,
        "vault_file": str(vault.path),
    }

    def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            keys = list(rows[0].keys()) if rows else ["epoch"]
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    metrics_path = out_dir / f"{run_id}_metrics.csv"
    continuity_path = out_dir / f"{run_id}_continuity.csv"
    branch_path = out_dir / f"{run_id}_branches.csv"
    _write_csv(metrics_path, metrics_rows)
    _write_csv(continuity_path, continuity_rows)
    _write_csv(branch_path, branch_rows)

    summary_path = out_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    bridge_payload = {
        "network_lt": network_lt,
        "continuity": continuity_rows[-1] if continuity_rows else {},
        "training": {"baseline": baseline_score, "score": training_score, "delta": training_score - baseline_score},
    }
    bridge_path = reporter.generate_bridge_export(bridge_payload, output_dir=cfg.out_dir, run_id=run_id)
    report_path = reporter.generate_report(summary, metrics_rows, output_dir=cfg.out_dir)
    gabriel_path = reporter.generate_gabriel_summary(summary, metrics_rows, output_dir=cfg.out_dir)

    bridge.store_simulation_run({"summary": summary, "bridge": bridge_payload})
    return {
        "summary": summary,
        "paths": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "bridge": bridge_path,
            "report": report_path,
            "gabriel": gabriel_path,
            "continuity": str(continuity_path),
            "branches": str(branch_path),
            "vault": str(vault.path),
        },
    }


def run_training(cfg: SimConfig) -> Dict[str, object]:
    runs = max(1, int(cfg.training_runs))
    baseline = 0.0
    histories: List[Dict[str, float | int | str]] = []
    final_result: Dict[str, object] = {}
    for i in range(runs):
        result = run_once(cfg, run_index=i, baseline_score=baseline)
        score = float(result["summary"]["continuity_score"])
        histories.append({
            "run_index": i,
            "run_id": str(result["summary"]["run_id"]),
            "continuity_score": score,
            "baseline": baseline,
            "delta": score - baseline,
        })
        baseline = max(baseline, score)
        final_result = result

    out_dir = Path(cfg.out_dir)
    training_path = out_dir / f"{cfg.run_id}_training.json"
    training_summary = {
        "run_id": cfg.run_id,
        "training_runs": runs,
        "best_continuity_score": baseline,
        "history": histories,
        "monotonic_non_decreasing": all(h["delta"] >= -1e-9 for h in histories[1:]),
    }
    training_path.write_text(json.dumps(training_summary, indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== ANC v0.3 simulation complete ===")
    print(f"run_id: {cfg.run_id}")
    print(f"training runs: {runs}")
    print(f"best continuity score: {baseline:.6f}")
    print(f"training summary: {training_path}")
    return {"final": final_result, "training": training_summary, "training_path": str(training_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ANC v0.3 continuity simulation")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--run-id", type=str, default="anchor_sim_v0_3")
    parser.add_argument("--seed", type=int, default=__seed__)
    parser.add_argument("--validators", type=int, default=120)
    parser.add_argument("--delegators", type=int, default=4000)
    parser.add_argument("--cartel", action="store_true")
    parser.add_argument("--sybil", action="store_true")
    parser.add_argument("--continuity-window", type=int, default=36)
    parser.add_argument("--training-runs", type=int, default=1)
    args = parser.parse_args()

    cfg = SimConfig(
        epochs=args.epochs,
        run_id=args.run_id,
        seed=args.seed,
        n_validators=args.validators,
        n_delegators=args.delegators,
        enable_cartel=args.cartel,
        enable_sybil=args.sybil,
        continuity_window=max(3, args.continuity_window),
        training_runs=max(1, args.training_runs),
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
