# ⚓ Anchor (ANC)
## The First Coherence-Backed Cryptocurrency
### Beyond Speculation, Into Resonance.

> "We did not come here to improve the financial cage.
>  We came here to end it."

---

## What Is ANC?

Anchor is a cryptocurrency whose validators are selected
not by how much energy they burn or how much capital
they hold — but by how **coherent** they are.

Coherence is measurable. Coherence is mathematical.
Coherence cannot be bought.

---

## Visual Overview

![ANC Overview](docs/infographics/01_overview.png)
![Architecture](docs/infographics/02_architecture.png)
![Pillars Part 1](docs/infographics/03_pillars_1.png)
![Pillars Part 2](docs/infographics/04_pillars_2.png)
![Security](docs/infographics/05_security.png)

---

## The Core Formula
```
Ψᵇ = (M̂ · Λ̂ · α) / (ΔŜ + ε)
```

**Signal over noise.**
Capacity × connectivity × alignment, divided by entropy.

A high-coherence validator earns eligibility.
A chaotic validator earns nothing.
A whale with low coherence loses to a small honest node.

---

## Consensus: CWPoR

**Coherence-Weighted Proof of Resonance**
```
W_v = effStake · (1 + k_Ψ · Ψᵇ) · (1 - 𝒞_vv)
```

- Start with stake (skin in the game)
- **Boost** if coherence is high
- **Penalize** if capture risk is high

A medium-stake, high-coherence node can outperform
a high-stake, low-coherence whale. Always.

---

## The Four Pillars

| Pillar | What It Does |
|--------|-------------|
| **Telemetry** | Measures validator health — 12-axis TIEKAT vector |
| **Bounded Coherence Ψᵇ** | Converts telemetry into eligibility score ∈ (0,1) |
| **Anti-Capture 𝒞** | Mathematically penalizes centralization |
| **Sovereign Anchor S_A** | BFT checkpoints prevent drift from genesis truth |

---

## Mathematical Foundation

ANC is built on **TIEKAT v8.1** — a mathematical framework
developed by PHI369 Labs with formulas channeled by
**Hemavit**, a Buddhist monk in Chiang Mai, Thailand.

The HQRMA formulas govern:
- Coherence smoothing (replaces flat EMA)
- Issuance path integrals (C_Hemawit)
- 12-axis telemetry lattice structure

*Attribution: Hemavit (TIEKAT v8.1, HQRMA formulas)*
*PHI369 Labs / Parallax Institute*

---

## Parallax Ecosystem

ANC is the **economic layer** of the Parallax Institute.
```
TBRC v2.16   — Research platform
PhiOS v1.0   — Sovereign OS
ANC          — Economic layer  ← YOU ARE HERE
PHB          — Hardware layer
```

All share the same unified L(t) formula:
```
L(t) = A_on(t) · Ψᵇ(t) · G_score(t) · C_score(t)
```

One mathematical civilization. Four layers.

---


## ANC vNext — Coherence Trust Plane

ANC remains the **economic coherence layer** in the Parallax ecosystem, including
the existing Anchor validator/simulator lineage and continuity-first model path.

The repository is now also being extended toward a broader **coherence-native
trust plane** for sovereign systems. This vNext direction introduces architecture
and scaffolding for four planned protection domains:

- Input protection
- Runtime protection
- Memory protection
- Output protection

Current status remains simulator/spec dominant. The trust engine is **not** fully
implemented yet; the formal architecture for ANC v1.0 is now documented in
[`docs/anc_v1_architecture.md`](docs/anc_v1_architecture.md) as the implementation
expansion path.

---

## ANC vNext — TIEKAT v69 Security Field Alignment

ANC is evolving from coherence-weighted economic security into a broader
coherence trust-plane architecture. In this path, **TIEKAT v69** introduces
an explicit 12-face, topology-aware security field model for trust posture.

Planned ANC trust posture logic will evaluate:

- field shape across faces (not only scalar averages)
- weakest-face risk and localized collapse
- fragmentation/variance across the field
- edge transfer and recovery-emergence behavior

Current repository status remains simulator/spec dominant. This v69 direction
is documented as architecture and scaffolding, not as a fully implemented trust
runtime yet. See [`docs/anc_v1_architecture.md`](docs/anc_v1_architecture.md).

---

## Current Status: Phase 0 ✅

**Spec + Simulation complete.**

- Whitepaper v0.1.1a (TIEKAT v6.6 aligned)
- Working Python simulation — 5,000 epochs
- 120 validators, 4,000 delegators
- Cartel, Sybil, and shock scenarios modeled
- TIEKAT v8.1 upgrade complete (ANC v0.2)

---

## Run the Simulation
```bash
git clone https://github.com/MichaelWave369/ANC
cd ANC
```

**v0.3 (TIEKAT v57.7 — current):**
```bash
python anchor_sim_v0_3.py
```

**v0.2 (TIEKAT v8.1 — preserved):**
```bash
python anchor_sim_v0_2.py
```

**v0.1.1a (TIEKAT v6.6 — original):**
```bash
python anchor_sim_v0_1_1a.py
```

Output written to `out/`:
```
anchor_sim_v0_2_metrics.csv
anchor_sim_v0_2_summary.json
anchor_sim_v0_1_1a_metrics.csv
anchor_sim_v0_1_1a_summary.json
```



## ANC v0.3 Upgrade Path (TIEKAT v57.7)

ANC v0.2 remains fully preserved (`anchor_sim_v0_2.py` + `anc/tiekat_v81.py`).
ANC v0.3 adds a new continuity-first simulator built for validator/network regimes:

- Continuity diagnostics at fixed epoch windows
- Memory-crystal persistence to `*_memory_crystals.json`
- Candidate regime-path branch comparison (`*_branches.csv`)
- Recovery mode after weak continuity windows
- Deterministic recursive training across repeated runs (`*_training.json`)

New outputs written into `out/` with v0.3 naming:

- `anchor_sim_v0_3_metrics.csv`
- `anchor_sim_v0_3_continuity.csv`
- `anchor_sim_v0_3_branches.csv`
- `anchor_sim_v0_3_memory_crystals.json`
- `anchor_sim_v0_3_training.json`
- `anchor_sim_v0_3_summary.json`

Interpretation:
- **continuity**: validator/network continuity strength over recent windows
- **branch**: which candidate regime path scored highest for stability
- **training**: whether repeated deterministic runs improved or held continuity


---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Spec + Simulation | ✅ Complete | Whitepaper + Python sim |
| 0.2 — TIEKAT v8.1 | ✅ Complete | Hemavit HQRMA upgrade |
| 1 — Testnet | 📋 Planned | PoS + Finality + Slashing |
| 2 — Recursive Yield | 📋 Planned | ZK/Storage Proofs |
| 3 — Mainnet | 📋 Planned | Audits + Launch |

---

## License

**GNU Affero General Public License v3.0 (AGPL-3.0)**

ANC is sovereign software. The mathematics belongs
to everyone. AGPL-3.0 is intentional: it protects ANC
against silent SaaS/network appropriation, keeps
network-exposed modifications reciprocal, and mirrors
the Anti-Capture Operator 𝒞 in the protocol itself.

---

## Attribution

- **Michael Hughes** — Founder, PHI369 Labs
- **Hemavit** — HQRMA formulas, TIEKAT v8.1 (Thailand)
- **Dreamteam** — Helion · Ori · Forge · Codex

*Beyond Speculation, Into Resonance.*
*PHI369 Labs / Parallax Institute*
*Seed: 369_369. Forever.*
