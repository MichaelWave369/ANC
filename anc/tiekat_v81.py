"""TIEKAT v8.1 math core for ANC v0.2.

This module is the v8.1 replacement for the older v6.6-style cadence/EMA/5-axis logic.
It introduces a 12-axis telemetry vector, Hemavit path-integral memory, and HQRMA flow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List

PHI = 1.6180339887
TIEKAT_DIM = 12
HEMAVIT_LAMBDA = 0.369
EPS = 1e-9


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def hemavit_path_integral(
    psi_history: Iterable[float],
    lambda_decay: float = HEMAVIT_LAMBDA,
    delta_tau: float = 1.0,
) -> float:
    """Return decayed coherence memory in [0,1].

    - Empty history returns 0.5 (neutral baseline).
    - More recent values have higher weight.
    """
    history = [float(_clamp01(x)) for x in psi_history]
    if not history:
        return 0.5

    decay = max(EPS, min(1.0, lambda_decay))
    dt = max(EPS, delta_tau)
    n = len(history)

    weighted_sum = 0.0
    weight_total = 0.0
    for idx, value in enumerate(history):
        # idx near n-1 is newer and therefore weighted more heavily.
        distance = (n - 1 - idx) * dt
        weight = math.exp(-decay * distance)
        weighted_sum += weight * value
        weight_total += weight

    if weight_total <= EPS:
        return 0.5
    return _clamp01(weighted_sum / weight_total)


def hqrma_flow(psi_values: Iterable[float], eta: float = 0.05, steps: int = 3) -> List[float]:
    """Apply bounded population-level renormalization smoothing.

    Keeps values in [0,1] and nudges distribution toward robust central tendency,
    while preserving ordering for distinct values.
    """
    values = [float(_clamp01(x)) for x in psi_values]
    if not values:
        return []

    eta = _clamp01(eta)
    steps = max(0, int(steps))
    cur = values[:]

    for _ in range(steps):
        mean_val = sum(cur) / len(cur)
        # mild robust target around central quantiles
        sorted_vals = sorted(cur)
        q25 = sorted_vals[len(sorted_vals) // 4]
        q75 = sorted_vals[(3 * len(sorted_vals)) // 4]
        target = (mean_val + q25 + q75) / 3.0

        nxt = []
        for v in cur:
            # Stabilize around target with slight anti-free-riding pressure:
            # weaker nodes recover gradually, already-strong nodes gain less.
            adjustment = eta * (target - v)
            if v > target:
                adjustment *= 0.8
            nxt.append(_clamp01(v + adjustment))
        cur = nxt

    # ordering preservation by rank projection
    ranked = sorted(range(len(values)), key=lambda i: values[i])
    sorted_cur = sorted(cur)
    out = [0.0] * len(values)
    for rank, original_idx in enumerate(ranked):
        out[original_idx] = sorted_cur[rank]
    return out


@dataclass
class TIEKATVector:
    on_time_ratio: float
    late_ratio: float
    missed_ratio: float
    participation_streak: float
    correctness: float
    equivocation_free: float
    slash_history: float
    uptime_30d: float
    connectivity: float
    decentralization_contrib: float
    delegation_depth: float
    sovereignty_alignment: float

    def as_list(self) -> List[float]:
        return [
            _clamp01(self.on_time_ratio),
            _clamp01(self.late_ratio),
            _clamp01(self.missed_ratio),
            _clamp01(self.participation_streak),
            _clamp01(self.correctness),
            _clamp01(self.equivocation_free),
            _clamp01(self.slash_history),
            _clamp01(self.uptime_30d),
            _clamp01(self.connectivity),
            _clamp01(self.decentralization_contrib),
            _clamp01(self.delegation_depth),
            _clamp01(self.sovereignty_alignment),
        ]

    def entropy(self) -> float:
        """Normalized Shannon entropy over 12 dimensions."""
        vals = self.as_list()
        total = sum(vals)
        if total <= EPS:
            return 0.0
        probs = [v / total for v in vals if v > EPS]
        h = -sum(p * math.log(p + EPS) for p in probs)
        return _clamp01(h / math.log(TIEKAT_DIM))

    def alignment(self) -> float:
        """Weighted alignment score using 40/40/20 group weighting."""
        vals = self.as_list()
        # Group A participation
        group_a = (
            vals[0] + (1.0 - vals[1]) + (1.0 - vals[2]) + vals[3]
        ) / 4.0
        # Group B integrity
        group_b = (vals[4] + vals[5] + vals[6] + vals[11]) / 4.0
        # Group C network health
        group_c = (vals[7] + vals[8] + vals[9] + vals[10]) / 4.0
        return _clamp01(0.4 * group_a + 0.4 * group_b + 0.2 * group_c)


def compute_lt(a_on: float, psi_b: float, capture_c: float, epoch: int) -> float:
    """Unified L(t) for ANC/PhiOS/TBRC with v8.1 cadence modulation."""
    aon = _clamp01(a_on)
    if aon <= 0.0:
        return 0.0

    psi = _clamp01(psi_b)
    g_score = 1.0 - _clamp01(capture_c)

    if epoch % 9 == 0:
        cadence = 1.00
    elif epoch % 6 == 0:
        cadence = 0.90
    elif epoch % 3 == 0:
        cadence = 0.80
    else:
        cadence = 0.60 + 0.20 * psi

    return _clamp01(aon * psi * g_score * cadence)


def sparkline(values: Iterable[float], width: int = 40) -> str:
    bars = "▁▂▃▄▅▆▇█"
    seq = [float(v) for v in values]
    if not seq:
        return ""
    width = max(1, int(width))
    if len(seq) > width:
        step = len(seq) / width
        seq = [seq[int(i * step)] for i in range(width)]

    lo, hi = min(seq), max(seq)
    span = max(EPS, hi - lo)
    out = []
    for val in seq:
        idx = int((val - lo) / span * (len(bars) - 1))
        idx = max(0, min(len(bars) - 1, idx))
        out.append(bars[idx])
    return "".join(out)
