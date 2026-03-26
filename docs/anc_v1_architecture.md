# ANC v1.0 — Coherence Trust Plane Architecture (TIEKAT v69 Aligned)

## 1) Purpose

ANC v1.0 preserves ANC's simulator/economic lineage while extending ANC into a dual-domain architecture:

1. **Economic coherence domain (existing):** Anchor validator economics, anti-capture logic, and continuity-first simulation pathways.
2. **Security coherence domain (new):** trust/defense posture, anti-contamination controls, and policy enforcement for sovereign local-first systems.

ANC remains aligned with the PHI369 stack:

- **TIEKAT:** mathematical substrate and field geometry grammar.
- **ANC Trust Plane:** security interpretation, posture, and enforcement decisions.
- **PhiKernel:** runtime hooks where trust decisions are applied.
- **PhiOS:** observability surface for posture, incidents, and continuity state.

This document is architecture/specification guidance. It does **not** claim that the full v69 trust engine is already implemented.

## 2) Mission

ANC security-coherence mission is to protect sovereign systems against:

- jailbreak attacks
- prompt injection
- memory poisoning
- retrieval contamination
- runtime anomalies
- output leakage
- continuity corruption
- policy capture
- malware-like behavior in local-first execution contexts

## 3) Core Philosophy

Traditional security centers on perimeter and static rule sets.
ANC security centers on **coherent continuity over time and topology**.

Key framing:

- **Sovereignty:** decisions and controls remain inspectable and local.
- **Continuity:** trusted behavior remains stable across sessions and updates.
- **Anchor Integrity:** trusted baselines must stay measurable and recoverable.
- **Shape-aware security:** a high global average is insufficient if local regions collapse.

## 4) TIEKAT v69 Alignment

ANC trust posture is now modeled using a **12-face dodecahedron coherence field** rather than only scalar coherence.

Implication:

- Previous scalar security coherence remains useful as a summary.
- Primary trust logic moves to **face-aware and topology-aware analysis** of the security field.
- Posture transitions should account for field shape, not only global magnitude.

## 5) 12-Face Security Field Model

ANC v1.0 security state is represented over a 12-face field.

Conceptual components:

- `face_scores`: score per face in \([0,1]\)
- `weakest_face`: face index with minimum security score
- `field_variance`: spread of security across faces
- `field_balance`: balanced vs fragmented field condition
- `edge_flow`: transfer quality across adjacent faces
- `vertex_emergence`: emergent stability score from face/edge interactions

Security interpretation:

- **Localized degradation:** one/few faces collapse while global mean remains moderate.
- **System-wide degradation:** broad decline across most faces with weak edge support.

## 6) Security Shape vs Security Magnitude

A scalar average can hide critical weak points.

- **Security magnitude:** global average of face scores (useful but incomplete).
- **Security shape:** distribution and topology of scores across faces/edges/vertices.

Why this matters:

- A system with acceptable mean score can still be exploitable via one collapsed face.
- High variance often indicates fragmentation and potential cascade risk.
- Weak edge flow impairs containment and recovery propagation.

## 7) Four Protection Layers

### Input Guard

Evaluates prompts, retrieved chunks, and operator directives for adversarial content; maps detected pressure to one or more field faces.

### Runtime Guard

Monitors execution behavior, privilege boundaries, and command trajectories; updates face and edge risk states in real time.

### Memory Guard

Protects short/long-term memory channels from poisoning and continuity drift; tracks persistence-contamination topology.

### Output Guard

Evaluates outputs for leakage/policy breaches; measures whether risky patterns are localized or propagating across field topology.

## 8) Face-Aware Threat Mapping (Proposed)

This is an architectural proposal, not a final scientific claim.

Potential domain-to-face mappings may include:

- foundation / boot trust
- continuity / memory persistence
- integrity / tamper resistance
- logic / command correctness
- memory / poisoning resistance
- recovery / quarantine viability
- stability / sustained safe posture
- observer / operator/meta-detection

Remaining faces may represent additional policy/runtime/retrieval/output domains as calibration data matures.

## 9) Core Entities

### ThreatSignal

Normalized threat event with topology hints.

Example conceptual fields:

- `signal_type`, `source_layer`, `severity`, `confidence`, `timestamp`
- `face_targets` (candidate impacted faces)
- `edge_implications` (possible spread routes)

### CoherenceSecurityState

Shape-aware security state snapshot.

Example conceptual fields:

- `face_scores`
- `weakest_face`
- `field_variance`
- `edge_flow`
- `vertex_emergence`
- `posture`
- `anchor_integrity`

### GuardDecision

Policy-bound enforcement decision with topology rationale.

Example conceptual fields:

- `action`
- `rationale`
- `shape_risk_flags`
- `policy_trace`

### IncidentRecord

Immutable incident record for forensic continuity.

Example conceptual fields:

- `incident_id`
- `signals`
- `pre_state` / `post_state`
- `decision`
- `affected_faces`
- `continuity_delta`

## 10) Revised Metrics / Formulas (Architecture-Level)

These formulas are design notation for v1.0 planning, not production-validated claims.

### Security Field Vector

\[
\mathbf{F}_{sec}(t) = [f_1(t), f_2(t), \dots, f_{12}(t)], \quad f_i \in [0,1]
\]

### Scalar Summary (Optional)

\[
\bar{F}_{sec}(t) = \frac{1}{12}\sum_{i=1}^{12} f_i(t)
\]

### Weakest Security Face

\[
f_{\min}(t) = \min_i f_i(t), \qquad i_{\min}=\arg\min_i f_i(t)
\]

### Field Variance / Fragmentation

\[
\sigma_F^2(t)=\frac{1}{12}\sum_{i=1}^{12}(f_i(t)-\bar{F}_{sec}(t))^2
\]

Higher \(\sigma_F^2\) suggests fragmentation risk.

### Edge Transfer / Edge Flow

For adjacency set \(E\):

\[
\Phi_E(t)=\frac{1}{|E|}\sum_{(i,j)\in E} g_{ij}(t)
\]

Where \(g_{ij}(t)\) measures healthy transfer/containment capability along edge \((i,j)\).

### Contamination Topology

\[
\Lambda_{topo}(t)=h(\mathbf{F}_{sec}(t), E, \text{threat signals})
\]

Captures whether contamination is localized, edge-propagating, or field-wide.

### Recovery Vertex Score

For candidate recovery vertices \(V\):

\[
R_v(t)=\max_{k\in V} r_k(t)
\]

Where \(r_k\) estimates viable recovery emergence at vertex \(k\).

## 11) Shape-Aware Posture Model

ANC posture states remain:

- **SAFE**
- **WATCH**
- **DEGRADED**
- **HOSTILE**
- **QUARANTINED**

Posture transitions should consider:

- one-face collapse (very low \(f_{\min}\))
- high variance / fragmentation (high \(\sigma_F^2\))
- weak edge transfer (low \(\Phi_E\))
- unstable recovery emergence (low/volatile \(R_v\))

This prevents overreliance on only global-score thresholds.

## 12) Enforcement Actions

Available actions:

- `allow`
- `warn`
- `redact`
- `shadow`
- `sandbox`
- `quarantine`
- `refuse`
- `seal`

Shape-aware policy examples (architectural intent):

- one-face collapse + high confidence threat → `sandbox` / `quarantine`
- fragmentation with uncertain attribution → `warn` + `shadow` + targeted constraints
- strong contamination topology + low recovery vertex → `seal`

## 13) Integration Points

### TIEKAT Integration

- use v69 field language for face/edge/vertex security interpretation
- preserve scalar summary compatibility for dashboards and backward readability

### PhiKernel Hooks

- pre-execution guard scoring by face
- runtime updates for edge-flow degradation and propagation risk
- posture-driven enforcement checkpoints

### PhiOS View Model

- field-shape visualization (12-face health map)
- weakest-face and fragmentation indicators
- incident timeline with localized vs system-wide degradation flags

## 14) Proposed Repo Structure

```text
ANC/
├── anchor_sim_v0_1_1a.py
├── anchor_sim_v0_2.py
├── anchor_sim_v0_3.py
├── anc/
│   ├── __init__.py
│   ├── economic/
│   │   └── __init__.py
│   ├── core/
│   │   └── __init__.py
│   ├── detectors/
│   │   └── __init__.py
│   ├── guards/
│   │   └── __init__.py
│   ├── policy/
│   │   └── __init__.py
│   ├── integrations/
│   │   └── __init__.py
│   ├── continuity.py
│   ├── report.py
│   ├── parallax_bridge.py
│   ├── tiekat_v81.py
│   └── tiekat_v57.py
├── docs/
│   ├── anc_v1_architecture.md
│   └── infographics/
└── tests/
```

## 15) Detector Roadmap (v1.0)

Recommended initial detector families:

- jailbreak detector
- injection detector
- suspicious command/code detector
- memory contamination detector
- output leakage detector
- drift/anomaly detector

Future versions should include face-target calibration and topology propagation tagging per detector output.

## 16) Build Phases

### Phase 1: Core trust state engine

- implement topology-aware state models (`face_scores`, `weakest_face`, `field_variance`, `edge_flow`, `vertex_emergence`)
- define posture transition rules using shape-aware thresholds

### Phase 2: Detector integration

- attach detector outputs to face/edge threat mapping
- calibrate confidence and propagation semantics

### Phase 3: PhiKernel hook-in

- enforce shape-aware actions at runtime checkpoints
- support containment/recovery transitions

### Phase 4: PhiOS observatory surface

- visualize field shape, weakest-face risk, fragmentation, and recovery pathways
- support human-in-the-loop incident resolution

## 17) Success Criteria

ANC v1.0 succeeds when:

- economic simulator lineage remains intact and reproducible
- trust posture is explainable in face/edge/vertex terms
- one-face collapse and fragmentation are detected before broad compromise
- enforcement decisions are policy-traceable and topology-rationalized
- recovery workflows preserve continuity and anchor integrity

## 18) Immediate Next Step

Implement the **core topology-aware trust models and scoring engine** (still lightweight):

1. Define Python models for `ThreatSignal`, `CoherenceSecurityState`, `GuardDecision`, and `IncidentRecord` with face-aware fields.
2. Implement deterministic calculations for `face_scores`, `weakest_face`, `field_variance`, `edge_flow`, and `vertex_emergence` as architecture baselines.
3. Add posture-transition unit tests for one-face collapse, fragmented field, and weak-recovery scenarios.

No full detector/guard runtime implementation should be done before this model layer is stable.
