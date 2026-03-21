# ANC Changelog

Beyond Speculation, Into Resonance.
PHI369 Labs / Parallax Institute
Attribution: Hemavit (HQRMA, TIEKAT v8.1)

---


## v0.3.0 — TIEKAT v57.7 Continuity Upgrade (March 21, 2026)

- Added side-by-side `anchor_sim_v0_3.py` while preserving v0.2 files and runtime behavior.
- Added `anc/tiekat_v57.py` with deterministic continuity primitives (signatures, state vectors, memory crystals, vault, branch scoring).
- Added `anc/continuity.py` ANC-specific adapters for validator/network continuity scoring and crystal construction.
- Added continuity diagnostics windows, branch-path comparison, recovery mode, and recursive training outputs.
- Updated package metadata to `ANC v0.3.0` and `TIEKAT v57.7`.
- Updated report and bridge modules to export and summarize v0.3 continuity/training artifacts while remaining backward-compatible.
- Expanded tests for v0.2 compatibility and v57.7 continuity/vault/branch/training determinism.

## v0.2.0 — TIEKAT v8.1 Upgrade (in progress)

- Migration path from TIEKAT v6.6 to v8.1 implemented as additive side-by-side simulation (`anchor_sim_v0_2.py`), preserving v0.1.1a unchanged.
- Hemavit path integral introduced for coherence-memory issuance gating.
- HQRMA renormalization flow added as population-level coherence smoothing.
- 12-axis TIEKAT telemetry vector added as the primary validator telemetry model.
- Unified `L(t)` formulation aligned with ANC, PhiOS, and TBRC.
- Optional Parallax bridge added with graceful no-op behavior when PhiOS/TBRC are unavailable.
- Report artifacts expanded with simulation markdown report, bridge export, and Gabriel technical summary.
- Backward compatibility retained via feature flags for legacy-style telemetry/smoothing/issuance pathways.

## v0.1.1a — Initial Simulation (February 11, 2026)

First working CWPoR simulation. TIEKAT v6.6 aligned.

- Ψᵇ bounded coherence scoring
- Anti-Capture Operator 𝒞
- CWPoR consensus weight W_v
- Sovereign Anchor S_A drift bounds
- 369 cadence tiers (optional)
- Cartel, Sybil, shock scenario modeling
- Nakamoto + Gini coefficient tracking
- 5,000 epochs · 120 validators · 4,000 delegators
- Seed: 369_369
