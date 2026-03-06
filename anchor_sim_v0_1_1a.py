#!/usr/bin/env python3
"""
Anchor (ANC) — Sim Script v0.1.1a (TIEKAT v6.6–Aligned)

Adds to v0.1.1:
- Telemetry distribution p_v(k) -> entropy load ΔŜ
- Alignment α
- Quality Q̂, capacity M̂, coherence ratio Ψ, bounded Ψ^(b)
- Explicit capture operator 𝒞_vv
- Weight W = effStake * (1 + k_Ψ*Ψ^(b)) * (1 - 𝒞_vv)
- Issuance budget B_t = round(B_max * mean Ψ^(b)) and Permit gate

Outputs:
- out/<run_id>_metrics.csv
- out/<run_id>_summary.json
Optional:
- out/<run_id>_validators.csv snapshots
- plots if matplotlib enabled

Python 3.10+ only.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Config
# -----------------------------

@dataclass
class SimConfig:
    # Time / run length
    epochs: int = 5000
    seed: int = 369_369
    run_id: str = "anchor_sim_v0_1_1a"

    # Population
    n_validators: int = 120
    n_delegators: int = 4000
    initial_total_stake: float = 100_000_000.0

    validator_stake_lognorm_mu: float = 0.0
    validator_stake_lognorm_sigma: float = 1.0
    delegator_stake_lognorm_mu: float = -0.5
    delegator_stake_lognorm_sigma: float = 1.2

    # Fees & treasury
    fee_per_epoch: float = 2_000.0
    fee_burn_frac: float = 0.50
    validator_opex_per_epoch: float = 1.0

    # Issuance (TIEKAT gate)
    issuance_bmax_per_epoch: float = 12_000.0   # B_max in B_t=round(B_max*meanPsiB)
    psi_emit: float = 0.45                      # Ψ_emit threshold for Permit
    alpha_min: float = 0.55                     # α_min threshold for Permit
    anchor_on: int = 1                          # A_on(t) in Permit gate
    # If Permit=0, issuance is redirected to treasury
    redirect_to_treasury_on_no_permit: bool = True

    # Cadence tiers (optional; keep simple: disabled by default)
    enable_369_cadence_tiers: bool = False
    tier_small_mult: float = 0.75
    tier_med_mult: float = 1.00
    tier_large_mult: float = 1.35

    # EMA smoothing
    ema_alpha: float = 0.20

    # CWPoR bounded coherence influence
    k_psi: float = 0.10  # keep <= 0.20

    # 𝒞 diminishing returns thresholds (fractions of total bonded stake)
    c_on: bool = True
    t1_frac: float = 0.005
    t2_frac: float = 0.020

    # Delegation diversity bonus
    diversity_bonus_on: bool = True
    bonus_bottom_50: float = 1.06
    bonus_50_80: float = 1.03
    bonus_top_20: float = 1.00

    redelegate_every_epochs: int = 9
    redelegate_frac: float = 0.35
    strat_apr_chaser: float = 0.45
    strat_brand_inertia: float = 0.25
    strat_decentralizer: float = 0.20
    strat_balanced: float = 0.10

    # Operational quality model
    opq_mu: float = 0.85
    opq_sigma: float = 0.08
    required_attestations: int = 100

    # Fault injection (honest baseline)
    p_invalid_block: float = 0.0003
    p_equivocation: float = 0.00005

    # Shocks
    enable_shocks: bool = True
    shock_every_epochs: int = 333
    shock_fraction_validators: float = 0.18
    shock_duration_epochs: int = 3
    shock_severity_missed_multiplier: float = 3.5

    # Cartel scenario
    enable_cartel: bool = False
    cartel_fraction: float = 0.12
    cartel_commission_bps: int = 50
    non_cartel_commission_bps: int = 500
    cartel_brand_boost: float = 1.5
    cartel_secrecy_bias: float = 0.20  # cartel tends to be more opaque in sim (secrecy proxy)

    # Sybil scenario
    enable_sybil: bool = False
    sybil_count: int = 60
    sybil_stake_each: float = 25_000.0
    sybil_opq: float = 0.92

    # Capture operator 𝒞 parameters (v6.6 style)
    c_min: float = 0.02
    c_max: float = 0.65
    a_r: float = 0.30
    a_c: float = 0.15
    a_s: float = 0.20
    a_p: float = 0.25
    a_i: float = 0.10
    k_pow: float = 3.0  # for E(power)=1-exp(-k_pow*power)

    # Coherence computation params (v6.6)
    eps: float = 1e-9
    q_alpha: float = 0.60
    q_S: float = 0.40
    gamma_psi: float = 0.05

    # Optional coherence length proxy (Λ̂) — can keep constant if you prefer
    use_lambda_proxy: bool = True
    lambda_ref: float = 2.5

    # Slashing/jail
    slash_invalid_block_pct: float = 0.02
    slash_equivocation_pct: float = 0.10
    jail_invalid_epochs: int = 27
    jail_equivocation_epochs: int = 111

    # Attack cost proxy
    anc_price: float = 1.0
    impact: float = 6.0
    impact_power: float = 1.6

    # Outputs
    write_validator_csv_every: int = 0
    out_dir: str = "out"
    enable_plots: bool = False


# -----------------------------
# Helpers
# -----------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def gini(values: List[float]) -> float:
    vals = [v for v in values if v >= 0]
    if not vals:
        return 0.0
    s = sum(vals)
    if s == 0:
        return 0.0
    vals.sort()
    n = len(vals)
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2.0 * cum) / (n * s) - (n + 1) / n

def nakamoto_coefficient(weights: List[float], threshold: float) -> int:
    w = [max(0.0, x) for x in weights]
    total = sum(w)
    if total <= 0:
        return 0
    w.sort(reverse=True)
    target = threshold * total
    acc = 0.0
    k = 0
    for x in w:
        acc += x
        k += 1
        if acc >= target:
            return k
    return k

def percentile(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1

def topk_share(values: List[float], k: int) -> float:
    vals = [max(0.0, v) for v in values]
    total = sum(vals)
    if total <= 0:
        return 0.0
    vals.sort(reverse=True)
    return sum(vals[:min(k, len(vals))]) / total

def cost_to_buy_stake(amount_stake: float, cfg: SimConfig) -> float:
    x = max(0.0, amount_stake)
    # Nonlinear impact proxy (bounded)
    return cfg.anc_price * x * (1.0 + cfg.impact * (x ** cfg.impact_power) / (1.0 + (x ** cfg.impact_power)))


# -----------------------------
# Entities
# -----------------------------

@dataclass
class Validator:
    vid: int
    stake_bonded: float
    stake_delegated: float
    commission_bps: int
    opq: float
    psi_b_raw_last: float = 0.0
    psi_b_smooth: float = 0.55

    # Telemetry-derived fields (epoch-level)
    deltaS_hat: float = 0.0
    alpha: float = 0.0
    Q_hat: float = 0.0
    Lambda_hat: float = 0.5
    M_hat: float = 0.0
    psi_ratio: float = 0.0
    capture_C: float = 0.0

    jailed_until_epoch: int = -1
    is_cartel: bool = False
    is_sybil: bool = False

    profit_accum: float = 0.0
    slashed_total: float = 0.0
    bonded_initial: float = 0.0

@dataclass
class Delegator:
    did: int
    stake: float
    assigned_vid: int
    strategy: str


# -----------------------------
# 𝒞 diminishing returns (effective stake)
# -----------------------------

def effective_stake(stake_total: float, total_bonded: float, cfg: SimConfig) -> float:
    if not cfg.c_on:
        return stake_total
    T1 = cfg.t1_frac * total_bonded
    T2 = cfg.t2_frac * total_bonded
    if stake_total <= T1:
        return stake_total
    if stake_total <= T2:
        return T1 + 0.50 * (stake_total - T1)
    return T1 + 0.50 * (T2 - T1) + 0.20 * (stake_total - T2)


# -----------------------------
# TIEKAT v6.6 coherence and capture
# -----------------------------

def entropy_load_from_p(p: List[float], cfg: SimConfig) -> float:
    # H = - Σ p log(p+eps); ΔŜ = H/logK
    K = max(1, len(p))
    H = 0.0
    for x in p:
        if x > 0.0:
            H -= x * math.log(x + cfg.eps)
    return clip(H / (math.log(K) + cfg.eps), 0.0, 1.0)

def bounded_power_exp(power: float, cfg: SimConfig) -> float:
    # E(power) = 1 - exp(-k_pow * power)
    power = clip(power, 0.0, 1.0)
    return 1.0 - math.exp(-cfg.k_pow * power)

def capture_operator(risk: float, cent: float, secrecy: float, power: float, insul: float, cfg: SimConfig) -> float:
    C = (cfg.c_min
         + cfg.a_r * clip(risk, 0.0, 1.0)
         + cfg.a_c * clip(cent, 0.0, 1.0)
         + cfg.a_s * clip(secrecy, 0.0, 1.0)
         + cfg.a_p * bounded_power_exp(power, cfg)
         + cfg.a_i * clip(insul, 0.0, 1.0))
    return clip(C, cfg.c_min, cfg.c_max)

def compute_lambda_proxy(validators: Dict[int, Validator], cfg: SimConfig) -> float:
    # Proxy λ2 based on decentralization and operational quality.
    # Higher decentralization and higher mean opq -> better connectivity proxy.
    totals = [v.stake_bonded + v.stake_delegated for v in validators.values()]
    top5 = topk_share(totals, 5)
    mean_opq = statistics.fmean([v.opq for v in validators.values()])
    # λ2_proxy in (0,1], larger is better connectivity (smaller Λ̂)
    lam = clip(0.10 + 0.70 * (1.0 - top5) + 0.20 * mean_opq, 0.05, 1.0)
    return lam

def compute_psi_b_for_validator(p_dist: List[float], alpha: float, Lambda_hat: float, cfg: SimConfig) -> Tuple[float, float, float, float]:
    # Returns (psi_b_raw, Q_hat, M_hat, psi_ratio)
    deltaS_hat = entropy_load_from_p(p_dist, cfg)
    alpha = clip(alpha, 0.0, 1.0)

    Q_hat = clip(cfg.q_alpha * alpha + cfg.q_S * (1.0 - deltaS_hat), 0.0, 1.0)
    M_hat = clip(Q_hat / (Lambda_hat + cfg.eps), 0.0, 1.0)
    psi_ratio = (M_hat * Lambda_hat * alpha) / (deltaS_hat + cfg.eps)

    psi_anch = psi_ratio * (1.0 + cfg.gamma_psi)
    psi_b = psi_anch / (1.0 + psi_anch)  # bounded (0,1)
    return clip(psi_b, 0.0, 1.0), Q_hat, M_hat, psi_ratio

def update_ema(prev: float, x: float, alpha: float) -> float:
    return clamp01(alpha * x + (1.0 - alpha) * prev)


# -----------------------------
# Shocks
# -----------------------------

def maybe_apply_shock(epoch: int, validators: Dict[int, Validator], shock_state: Dict, cfg: SimConfig) -> None:
    if not cfg.enable_shocks:
        return
    if epoch % cfg.shock_every_epochs == 0:
        vids = list(validators.keys())
        random.shuffle(vids)
        k = max(1, int(cfg.shock_fraction_validators * len(vids)))
        shock_state["active_until"] = epoch + cfg.shock_duration_epochs
        shock_state["affected"] = set(vids[:k])

def is_shocked(epoch: int, vid: int, shock_state: Dict) -> bool:
    if "active_until" not in shock_state:
        return False
    if epoch >= shock_state["active_until"]:
        return False
    return vid in shock_state.get("affected", set())


# -----------------------------
# Initialization
# -----------------------------

def init_validators(cfg: SimConfig) -> Dict[int, Validator]:
    random.seed(cfg.seed)
    raw = [random.lognormvariate(cfg.validator_stake_lognorm_mu, cfg.validator_stake_lognorm_sigma)
           for _ in range(cfg.n_validators)]
    s_raw = sum(raw)
    total_bonded = 0.60 * cfg.initial_total_stake
    scale = total_bonded / s_raw if s_raw > 0 else 1.0

    validators: Dict[int, Validator] = {}
    for i in range(cfg.n_validators):
        bonded = raw[i] * scale
        opq = clamp01(random.gauss(cfg.opq_mu, cfg.opq_sigma))
        v = Validator(
            vid=i,
            stake_bonded=bonded,
            stake_delegated=0.0,
            commission_bps=500,
            opq=opq,
            psi_b_smooth=clamp01(random.uniform(0.45, 0.75)),
        )
        v.bonded_initial = bonded
        validators[i] = v

    if cfg.enable_cartel:
        vids = list(validators.keys())
        random.shuffle(vids)
        ccount = max(1, int(cfg.cartel_fraction * len(vids)))
        cartel_set = set(vids[:ccount])
        for vid, v in validators.items():
            v.is_cartel = (vid in cartel_set)
            v.commission_bps = cfg.cartel_commission_bps if v.is_cartel else cfg.non_cartel_commission_bps

    if cfg.enable_sybil:
        start = cfg.n_validators
        for j in range(cfg.sybil_count):
            vid = start + j
            v = Validator(
                vid=vid,
                stake_bonded=cfg.sybil_stake_each,
                stake_delegated=0.0,
                commission_bps=800,
                opq=cfg.sybil_opq,
                psi_b_smooth=clamp01(random.uniform(0.45, 0.70)),
                is_sybil=True,
            )
            v.bonded_initial = v.stake_bonded
            validators[vid] = v

    return validators

def init_delegators(validators: Dict[int, Validator], cfg: SimConfig) -> List[Delegator]:
    raw = [random.lognormvariate(cfg.delegator_stake_lognorm_mu, cfg.delegator_stake_lognorm_sigma)
           for _ in range(cfg.n_delegators)]
    s_raw = sum(raw)
    total_del = 0.40 * cfg.initial_total_stake
    scale = total_del / s_raw if s_raw > 0 else 1.0

    vids = list(validators.keys())
    bonded = [validators[vid].stake_bonded for vid in vids]
    total_bonded = sum(bonded)
    probs = [b / total_bonded if total_bonded > 0 else 1.0 / len(vids) for b in bonded]

    def draw_vid() -> int:
        r = random.random()
        acc = 0.0
        for vid, p in zip(vids, probs):
            acc += p
            if r <= acc:
                return vid
        return vids[-1]

    def draw_strategy() -> str:
        r = random.random()
        a = cfg.strat_apr_chaser
        b = a + cfg.strat_brand_inertia
        c = b + cfg.strat_decentralizer
        d = c + cfg.strat_balanced
        if r < a:
            return "apr"
        if r < b:
            return "brand"
        if r < c:
            return "decent"
        if r < d:
            return "balanced"
        return "random"

    delegators: List[Delegator] = []
    for i in range(cfg.n_delegators):
        stake = raw[i] * scale
        vid = draw_vid()
        delegators.append(Delegator(did=i, stake=stake, assigned_vid=vid, strategy=draw_strategy()))
        validators[vid].stake_delegated += stake

    return delegators


# -----------------------------
# Delegation behavior
# -----------------------------

def compute_validator_rankings(validators: Dict[int, Validator]) -> Tuple[List[int], List[int]]:
    items = []
    for vid, v in validators.items():
        total = v.stake_bonded + v.stake_delegated
        items.append((total, vid))
    items.sort()
    asc = [vid for _, vid in items]
    desc = list(reversed(asc))
    return asc, desc

def delegation_bonus_multiplier(vid: int, asc: List[int], cfg: SimConfig) -> float:
    if not cfg.diversity_bonus_on:
        return 1.0
    n = len(asc)
    if n == 0:
        return 1.0
    rank = asc.index(vid)  # ok for sim sizes
    pct = 100.0 * (rank + 1) / n
    if pct <= 50.0:
        return cfg.bonus_bottom_50
    if pct <= 80.0:
        return cfg.bonus_50_80
    return cfg.bonus_top_20

def estimate_validator_attractiveness(validators: Dict[int, Validator], cfg: SimConfig) -> Dict[int, float]:
    total_bonded = sum(v.stake_bonded for v in validators.values())
    y = {}
    for vid, v in validators.items():
        total = v.stake_bonded + v.stake_delegated
        eff = effective_stake(total, total_bonded, cfg)
        # Use bounded coherence and capture dampening
        w = eff * (1.0 + cfg.k_psi * v.psi_b_smooth) * (1.0 - v.capture_C)
        net = w * (1.0 - v.commission_bps / 10000.0)
        y[vid] = max(0.0, net)
    return y

def redelegate(delegators: List[Delegator], validators: Dict[int, Validator], epoch: int, cfg: SimConfig) -> None:
    if cfg.redelegate_every_epochs <= 0:
        return
    if epoch % cfg.redelegate_every_epochs != 0:
        return

    movers = random.sample(delegators, k=max(1, int(cfg.redelegate_frac * len(delegators))))
    asc, desc = compute_validator_rankings(validators)
    attr = estimate_validator_attractiveness(validators, cfg)

    top_by_attr = sorted(attr.items(), key=lambda kv: kv[1], reverse=True)
    top_vids = [vid for vid, _ in top_by_attr[:max(10, len(validators)//10)]]
    bottom_half = set(asc[:max(1, len(asc)//2)])
    top_large = desc[:max(10, len(desc)//10)]

    for d in movers:
        cur = d.assigned_vid
        validators[cur].stake_delegated = max(0.0, validators[cur].stake_delegated - d.stake)

        if d.strategy == "apr":
            candidates = top_vids[:]
            random.shuffle(candidates)
            d.assigned_vid = max(candidates, key=lambda vid: attr.get(vid, 0.0))

        elif d.strategy == "brand":
            candidates = top_large[:]
            def brand_score(vid: int) -> float:
                base = attr.get(vid, 0.0)
                if cfg.enable_cartel and validators[vid].is_cartel:
                    base *= cfg.cartel_brand_boost
                return base * (0.90 + 0.20 * random.random())
            d.assigned_vid = max(candidates, key=brand_score)

        elif d.strategy == "decent":
            candidates = list(bottom_half)
            random.shuffle(candidates)
            pool = candidates[:min(len(candidates), 50)]
            d.assigned_vid = max(pool, key=lambda vid: attr.get(vid, 0.0))

        elif d.strategy == "balanced":
            def balanced_score(vid: int) -> float:
                v = validators[vid]
                total = v.stake_bonded + v.stake_delegated
                size_pref = 1.0 / math.sqrt(1.0 + total / 1_000_000.0)
                return attr.get(vid, 0.0) * (0.75 + 0.25 * v.psi_b_smooth) * size_pref
            candidates = top_vids + random.sample(list(validators.keys()), k=min(25, len(validators)))
            d.assigned_vid = max(candidates, key=balanced_score)

        else:
            d.assigned_vid = random.choice(list(validators.keys()))

        validators[d.assigned_vid].stake_delegated += d.stake


# -----------------------------
# Epoch simulation
# -----------------------------

def simulate_epoch(epoch: int,
                   validators: Dict[int, Validator],
                   delegators: List[Delegator],
                   treasury: Dict[str, float],
                   shock_state: Dict,
                   cfg: SimConfig) -> Dict[str, float]:
    maybe_apply_shock(epoch, validators, shock_state, cfg)

    # Global lambda proxy (scheduled each epoch for simplicity)
    if cfg.use_lambda_proxy:
        lam2 = compute_lambda_proxy(validators, cfg)
        Lambda_hat_global = clip((1.0 / math.sqrt(lam2 + cfg.eps)) * (1.0 / cfg.lambda_ref), 0.0, 1.0)
    else:
        Lambda_hat_global = 0.50

    # 1) Telemetry -> ΔŜ, α -> Ψ^(b); compute capture 𝒞
    total_stake_all = sum(v.stake_bonded + v.stake_delegated for v in validators.values()) + cfg.eps
    total_bonded = sum(v.stake_bonded for v in validators.values()) + cfg.eps

    # For centrality/power proxies we use stake share
    for vid, v in validators.items():
        # If jailed, we still compute a low coherence (no participation)
        if epoch < v.jailed_until_epoch:
            # p distribution concentrates on "missed"
            p = [0.0, 0.0, 1.0, 0.0, 0.0]  # [on_time, late, missed, invalid, equivocation]
            v.deltaS_hat = entropy_load_from_p(p, cfg)
            v.alpha = 0.0
            v.Lambda_hat = Lambda_hat_global
            psi_b_raw, v.Q_hat, v.M_hat, v.psi_ratio = compute_psi_b_for_validator(p, v.alpha, v.Lambda_hat, cfg)
            v.psi_b_raw_last = psi_b_raw
            v.psi_b_smooth = update_ema(v.psi_b_smooth, psi_b_raw, cfg.ema_alpha)
        else:
            req = cfg.required_attestations

            # Base missed/late rates from opq
            base_miss_p = clamp01(0.20 * (1.0 - v.opq))
            base_late_p = clamp01(0.12 * (1.0 - v.opq))
            if is_shocked(epoch, vid, shock_state):
                base_miss_p = clamp01(base_miss_p * cfg.shock_severity_missed_multiplier)

            missed = sum(1 for _ in range(req) if random.random() < base_miss_p)
            submitted = req - missed
            late = sum(1 for _ in range(submitted) if random.random() < base_late_p)
            on_time = submitted - late

            invalid_block = (random.random() < cfg.p_invalid_block)
            equivocation = (random.random() < cfg.p_equivocation)

            # Build p_v(k) distribution over K=5 outcomes (simple)
            # Categories: on_time, late, missed, invalid_evidence, equivocation_evidence
            total_events = req + 2  # include evidence slots for normalization stability
            p = [
                on_time / total_events,
                late / total_events,
                missed / total_events,
                (1.0 / total_events) if invalid_block else 0.0,
                (1.0 / total_events) if equivocation else 0.0,
            ]
            # Normalize exactly
            s = sum(p)
            p = [x / (s + cfg.eps) for x in p]

            # Alignment α from performance: (on_time ratio + correctness)
            U = on_time / max(1, req)
            punctuality = on_time / max(1, on_time + late)
            correctness = 0.0 if (invalid_block or equivocation) else 1.0
            align_v = 0.55 * U + 0.25 * punctuality + 0.20 * correctness
            v.alpha = clip(align_v, 0.0, 1.0)

            v.deltaS_hat = entropy_load_from_p(p, cfg)
            v.Lambda_hat = Lambda_hat_global

            psi_b_raw, v.Q_hat, v.M_hat, v.psi_ratio = compute_psi_b_for_validator(p, v.alpha, v.Lambda_hat, cfg)
            v.psi_b_raw_last = psi_b_raw
            v.psi_b_smooth = update_ema(v.psi_b_smooth, psi_b_raw, cfg.ema_alpha)

            # Apply slashing/jail for faults
            if equivocation:
                slash_amt = cfg.slash_equivocation_pct * v.stake_bonded
                v.stake_bonded = max(0.0, v.stake_bonded - slash_amt)
                v.slashed_total += slash_amt
                v.jailed_until_epoch = epoch + cfg.jail_equivocation_epochs
                v.psi_b_smooth = 0.0
            elif invalid_block:
                slash_amt = cfg.slash_invalid_block_pct * v.stake_bonded
                v.stake_bonded = max(0.0, v.stake_bonded - slash_amt)
                v.slashed_total += slash_amt
                v.jailed_until_epoch = epoch + cfg.jail_invalid_epochs

        # Compute capture inputs (risk/cent/secrecy/power/insul)
        total = v.stake_bonded + v.stake_delegated
        power_share = clip(total / total_stake_all, 0.0, 1.0)
        power_v = clip(power_share / 0.10, 0.0, 1.0)  # 10% share maps to power=1

        cent_v = power_v  # simple proxy; replace with proposer share if you model leader election
        risk_v = clip(v.slashed_total / (v.bonded_initial + cfg.eps), 0.0, 1.0)

        # secrecy proxy: cartel more opaque; random minor opacity otherwise
        secrecy_v = clip((cfg.cartel_secrecy_bias if v.is_cartel else 0.05) + 0.05 * random.random(), 0.0, 1.0)

        # insulation proxy: isolation/correlation risk; lower opq implies higher insul
        insul_v = clip(0.60 * (1.0 - v.opq) + 0.10 * random.random(), 0.0, 1.0)

        v.capture_C = capture_operator(risk_v, cent_v, secrecy_v, power_v, insul_v, cfg)

    # 2) Delegation flow
    redelegate(delegators, validators, epoch, cfg)

    # 3) Fees
    burn = cfg.fee_burn_frac * cfg.fee_per_epoch
    tre = (1.0 - cfg.fee_burn_frac) * cfg.fee_per_epoch
    treasury["treasury"] += tre

    # 4) Issuance gate: B_t and Permit(t)
    mean_psi_b = statistics.fmean([v.psi_b_smooth for v in validators.values()])
    mean_alpha = statistics.fmean([v.alpha for v in validators.values()])

    # Cadence tiers (optional)
    cadence_mult = 1.0
    if cfg.enable_369_cadence_tiers:
        if epoch % 9 == 0:
            cadence_mult = cfg.tier_large_mult
        elif epoch % 6 == 0:
            cadence_mult = cfg.tier_med_mult
        elif epoch % 3 == 0:
            cadence_mult = cfg.tier_small_mult

    B_t = round(cfg.issuance_bmax_per_epoch * mean_psi_b * cadence_mult)

    permit = (1 if (mean_psi_b > cfg.psi_emit and mean_alpha > cfg.alpha_min and cfg.anchor_on == 1) else 0)

    if permit == 0:
        if cfg.redirect_to_treasury_on_no_permit:
            treasury["treasury"] += B_t
        B_t_distribute = 0
    else:
        B_t_distribute = B_t

    # 5) Reward distribution using W_v = effStake*(1+k*PsiB)*(1-C)
    if B_t_distribute > 0:
        total_bonded_now = sum(v.stake_bonded for v in validators.values()) + cfg.eps
        weights: Dict[int, float] = {}
        sumW = 0.0
        for vid, v in validators.items():
            if epoch < v.jailed_until_epoch:
                continue
            total = v.stake_bonded + v.stake_delegated
            eff = effective_stake(total, total_bonded_now, cfg)
            w = eff * (1.0 + cfg.k_psi * v.psi_b_smooth) * (1.0 - v.capture_C)
            w = max(0.0, w)
            weights[vid] = w
            sumW += w

        asc, _ = compute_validator_rankings(validators)

        del_by_vid: Dict[int, float] = {}
        for d in delegators:
            del_by_vid[d.assigned_vid] = del_by_vid.get(d.assigned_vid, 0.0) + d.stake

        if sumW <= 0:
            treasury["treasury"] += B_t_distribute
        else:
            for vid, v in validators.items():
                if vid not in weights:
                    continue
                R = B_t_distribute * (weights[vid] / sumW)
                R_net = max(0.0, R - cfg.validator_opex_per_epoch)

                commission = R_net * (v.commission_bps / 10000.0)
                distributable = R_net - commission
                v.profit_accum += commission

                delegated = del_by_vid.get(vid, 0.0)
                if delegated > 0:
                    bonus = delegation_bonus_multiplier(vid, asc, cfg)
                    payout = distributable * bonus
                    subsidy = max(0.0, payout - distributable)
                    treasury["treasury"] = max(0.0, treasury["treasury"] - subsidy)
                else:
                    v.profit_accum += distributable

    # 6) Metrics
    stake_totals = [v.stake_bonded + v.stake_delegated for v in validators.values()]
    bonded = [v.stake_bonded for v in validators.values()]
    eff_stakes = [effective_stake(st, sum(bonded)+cfg.eps, cfg) for st in stake_totals]
    psib = [v.psi_b_smooth for v in validators.values()]
    Ccaps = [v.capture_C for v in validators.values()]

    jailed = sum(1 for v in validators.values() if epoch < v.jailed_until_epoch)

    total_bonded_now = sum(bonded)
    need_halt = (1.0 / 3.0) * total_bonded_now
    need_take = (2.0 / 3.0) * total_bonded_now
    cost_halt = cost_to_buy_stake(need_halt, cfg)
    cost_take = cost_to_buy_stake(need_take, cfg)

    metrics = {
        "epoch": float(epoch),
        "total_stake": sum(stake_totals),
        "total_bonded": total_bonded_now,
        "treasury": treasury["treasury"],
        "permit": float(permit),
        "B_t": float(B_t),
        "B_t_distribute": float(B_t_distribute),
        "mean_psib": mean_psi_b,
        "mean_alpha": mean_alpha,
        "avg_C": statistics.fmean(Ccaps) if Ccaps else 0.0,
        "p10_psib": percentile(psib, 10),
        "p50_psib": percentile(psib, 50),
        "p90_psib": percentile(psib, 90),
        "pct_below_psi_emit": sum(1 for p in psib if p < cfg.psi_emit) / max(1, len(psib)),
        "jailed_validators": float(jailed),
        "gini_stake": gini(stake_totals),
        "gini_rewards_proxy": gini([v.profit_accum for v in validators.values()]),
        "nakamoto_33_stake": float(nakamoto_coefficient(stake_totals, 1/3)),
        "nakamoto_33_effstake": float(nakamoto_coefficient(eff_stakes, 1/3)),
        "nakamoto_67_stake": float(nakamoto_coefficient(stake_totals, 2/3)),
        "nakamoto_67_effstake": float(nakamoto_coefficient(eff_stakes, 2/3)),
        "top1_share": topk_share(stake_totals, 1),
        "top5_share": topk_share(stake_totals, 5),
        "top10_share": topk_share(stake_totals, 10),
        "cost_halt_finality_proxy": cost_halt,
        "cost_takeover_proxy": cost_take,
        "shock_active": 1.0 if (cfg.enable_shocks and epoch < shock_state.get("active_until", -1)) else 0.0,
        "k_psi": cfg.k_psi,
        "c_on": 1.0 if cfg.c_on else 0.0,
    }
    return metrics


# -----------------------------
# IO
# -----------------------------

def ensure_out_dir(cfg: SimConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

def write_metrics_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_validators_snapshot_csv(path: str, validators: Dict[int, Validator], epoch: int) -> None:
    fieldnames = [
        "epoch","vid","stake_bonded","stake_delegated","commission_bps","opq",
        "deltaS_hat","alpha","Q_hat","Lambda_hat","M_hat","psi_ratio",
        "psi_b_raw_last","psi_b_smooth","capture_C","jailed_until_epoch",
        "profit_accum","slashed_total","is_cartel","is_sybil"
    ]
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for v in validators.values():
            w.writerow({
                "epoch": epoch,
                "vid": v.vid,
                "stake_bonded": v.stake_bonded,
                "stake_delegated": v.stake_delegated,
                "commission_bps": v.commission_bps,
                "opq": v.opq,
                "deltaS_hat": v.deltaS_hat,
                "alpha": v.alpha,
                "Q_hat": v.Q_hat,
                "Lambda_hat": v.Lambda_hat,
                "M_hat": v.M_hat,
                "psi_ratio": v.psi_ratio,
                "psi_b_raw_last": v.psi_b_raw_last,
                "psi_b_smooth": v.psi_b_smooth,
                "capture_C": v.capture_C,
                "jailed_until_epoch": v.jailed_until_epoch,
                "profit_accum": v.profit_accum,
                "slashed_total": v.slashed_total,
                "is_cartel": int(v.is_cartel),
                "is_sybil": int(v.is_sybil),
            })

def maybe_plot(metrics_path: str, cfg: SimConfig) -> None:
    if not cfg.enable_plots:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}")
        return

    epochs = []
    mean_psib = []
    permit = []
    n33_eff = []
    top5 = []
    avgC = []
    with open(metrics_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(float(row["epoch"])))
            mean_psib.append(float(row["mean_psib"]))
            permit.append(float(row["permit"]))
            n33_eff.append(float(row["nakamoto_33_effstake"]))
            top5.append(float(row["top5_share"]))
            avgC.append(float(row["avg_C"]))

    def simple_plot(x, y, title, ylabel, fname):
        plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.tight_layout()
        outp = os.path.join(cfg.out_dir, fname)
        plt.savefig(outp, dpi=160)
        plt.close()

    simple_plot(epochs, mean_psib, "Mean Ψ^(b) over time", "mean_psib", f"{cfg.run_id}_mean_psib.png")
    simple_plot(epochs, permit, "Permit(t) over time", "permit", f"{cfg.run_id}_permit.png")
    simple_plot(epochs, n33_eff, "Nakamoto (1/3) by effective stake", "nakamoto_33_effstake", f"{cfg.run_id}_nakamoto33_eff.png")
    simple_plot(epochs, top5, "Top-5 stake share", "top5_share", f"{cfg.run_id}_top5_share.png")
    simple_plot(epochs, avgC, "Average capture 𝒞", "avg_C", f"{cfg.run_id}_avgC.png")


# -----------------------------
# Run
# -----------------------------

def run_once(cfg: SimConfig) -> Dict[str, str]:
    ensure_out_dir(cfg)
    random.seed(cfg.seed)

    validators = init_validators(cfg)
    delegators = init_delegators(validators, cfg)

    treasury = {"treasury": 0.0}
    shock_state: Dict = {}

    metrics_rows: List[Dict[str, float]] = []
    validators_csv_path = os.path.join(cfg.out_dir, f"{cfg.run_id}_validators.csv") if cfg.write_validator_csv_every else ""

    start = time.time()
    for epoch in range(cfg.epochs):
        m = simulate_epoch(epoch, validators, delegators, treasury, shock_state, cfg)
        metrics_rows.append(m)
        if cfg.write_validator_csv_every and (epoch % cfg.write_validator_csv_every == 0):
            write_validators_snapshot_csv(validators_csv_path, validators, epoch)

    dur = time.time() - start

    metrics_path = os.path.join(cfg.out_dir, f"{cfg.run_id}_metrics.csv")
    write_metrics_csv(metrics_path, metrics_rows)

    last = metrics_rows[-1] if metrics_rows else {}
    summary = {
        "run_id": cfg.run_id,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "duration_sec": dur,
        "config": asdict(cfg),
        "final_metrics": last,
    }
    summary_path = os.path.join(cfg.out_dir, f"{cfg.run_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    maybe_plot(metrics_path, cfg)

    return {"metrics_csv": metrics_path, "summary_json": summary_path, "validators_csv": validators_csv_path}


def main():
    cfg = SimConfig()

    # --- Scenario presets ---
    # Baseline: defaults

    # Cartel attempt:
    # cfg.enable_cartel = True

    # Sybil attempt:
    # cfg.enable_sybil = True

    # Stronger issuance gate (stricter):
    # cfg.psi_emit = 0.55
    # cfg.alpha_min = 0.65

    # Enable cadence tiers:
    # cfg.enable_369_cadence_tiers = True

    # Plots:
    # cfg.enable_plots = True

    # Validator snapshots:
    # cfg.write_validator_csv_every = 100

    print(f"[run] {cfg.run_id} epochs={cfg.epochs} seed={cfg.seed}")
    out = run_once(cfg)
    print("[done]")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
