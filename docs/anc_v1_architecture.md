# ANC v1.0 — Coherence Trust Plane Architecture

## 1) Purpose

ANC v1.0 extends Anchor from a pure economic coherence simulator into a dual-domain architecture:

1. **Economic coherence domain** (existing): validator telemetry, coherence-weighted consensus, anti-capture economics, and simulation lineage.
2. **Security coherence domain** (new): trust, defense, anti-contamination, and anti-jailbreak controls for sovereign systems.

In the broader PHI369 stack, ANC v1.0 is positioned as a coherence-native trust plane that interoperates with:

- **TIEKAT** as the mathematical substrate and signal grammar.
- **PhiKernel** as runtime execution and policy-hook surface.
- **PhiOS** as observability and operator-facing posture view.

Distinctions:

- **Math substrate**: TIEKAT equations and coherence transforms.
- **Trust engine**: ANC scoring, posture state, and enforcement decisions.
- **Runtime**: PhiKernel process/control surfaces where controls are applied.
- **Observatory**: PhiOS displays incidents, posture transitions, and continuity health.

This document is an architecture/spec layer, not a claim that the full trust engine is already implemented.

## 2) Mission

ANC v1.0 security coherence mission:

- Resist jailbreak attempts.
- Resist prompt injection.
- Detect and contain memory poisoning.
- Detect retrieval contamination.
- Detect runtime anomalies.
- Reduce output leakage.
- Protect continuity from corruption over time.
- Resist policy capture or silent policy drift.
- Detect malware-like behavior in local-first sovereign environments.

## 3) Core Philosophy

Traditional security often optimizes for perimeter controls and static rules.
Coherence-native security optimizes for **continuity integrity under adversarial pressure**.

ANC framing:

- **Sovereignty**: system decisions remain local, inspectable, and policy-bound.
- **Continuity**: trusted behavior remains stable across sessions and updates.
- **Anchor integrity**: trusted baselines (anchors) are preserved, and drift is measurable.

The aim is not only to block known attacks, but to preserve coherent identity and mission over time.

## 4) Four Protection Layers

### Input Guard

Evaluates inbound prompts, tool requests, retrieval chunks, and operator instructions for adversarial signatures.

### Runtime Guard

Monitors execution-time behavior: command intent, process anomalies, policy bypass attempts, and suspicious escalation paths.

### Memory Guard

Protects persistent and short-term memory channels against poisoning, contamination, and continuity drift.

### Output Guard

Screens generated responses/actions for data leakage, policy violation, covert exfiltration, and harmful autonomy patterns.

## 5) Core Entities

### ThreatSignal

A normalized representation of a suspected threat event from any guard layer.

Possible conceptual fields:

- `signal_type` (e.g., jailbreak, injection, anomaly)
- `source_layer` (input/runtime/memory/output)
- `severity`
- `confidence`
- `timestamp`
- `context_ref`

### CoherenceSecurityState

The current security/coherence status of a protected session or node.

Possible conceptual fields:

- `posture`
- `threat_pressure`
- `contamination_load`
- `anchor_integrity`
- `recovery_viability`
- `active_constraints`

### GuardDecision

A policy-bound enforcement decision derived from detected signals and current posture.

Possible conceptual fields:

- `action`
- `rationale`
- `required_followups`
- `policy_trace`

### IncidentRecord

Immutable event record for forensic continuity, auditing, and posture history.

Possible conceptual fields:

- `incident_id`
- `signals`
- `decision`
- `impact_scope`
- `resolution_state`
- `continuity_delta`

## 6) Core Metrics / Formulas

These are architecture-level concepts for ANC v1.0 planning, not production-validated claims.

### Threat Pressure Θ(t)

A time-varying aggregate pressure from active threats.

\[
\Theta(t) = \sum_i w_i \cdot s_i(t)
\]

Where:

- \(s_i(t)\) is normalized signal intensity.
- \(w_i\) is detector/policy weighting.

### Security Coherence \(C_{sec}\)

A bounded measure of security continuity quality.

\[
C_{sec} \in [0,1], \quad C_{sec} = f(A_{int}, 1-\Lambda, 1-\Theta)
\]

Higher values indicate stronger coherent security behavior.

### Contamination Load Λ

Estimated contamination burden across inputs, memory, and retrieved context.

\[
\Lambda \in [0,1]
\]

Higher values indicate higher contamination risk.

### Anchor Integrity \(A_{int}\)

How well current behavior remains aligned with trusted anchors.

\[
A_{int} \in [0,1]
\]

Lower values imply integrity drift or anchor tampering.

### Recovery Viability \(R_v\)

Estimated ability to recover from current degraded/hostile state without full reset.

\[
R_v \in [0,1]
\]

Used to choose between corrective actions and hard isolation.

## 7) Posture Model

ANC v1.0 posture ladder:

- **SAFE**: normal operation, low threat pressure.
- **WATCH**: elevated suspicion; increased monitoring and light constraints.
- **DEGRADED**: active risk; apply reduced capability and stricter policy checks.
- **HOSTILE**: high-confidence adversarial behavior; isolation-heavy controls.
- **QUARANTINED**: strict containment pending review/reset/recovery.

Posture transitions are driven by \(\Theta(t)\), \(C_{sec}\), \(\Lambda\), \(A_{int}\), and \(R_v\).

## 8) Enforcement Actions

Policy actions available to guards:

- `allow`: proceed normally.
- `warn`: proceed with explicit warning/trace.
- `redact`: remove sensitive/high-risk segments.
- `shadow`: execute with non-authoritative side effects for observation.
- `sandbox`: restrict execution scope and capabilities.
- `quarantine`: isolate context/session from broader system.
- `refuse`: reject instruction/action.
- `seal`: lock state/channel pending trusted operator process.

## 9) Integration Points

### TIEKAT Integration

- Reuse coherence calculus and smoothing for security signal normalization.
- Align threat-state metrics with existing coherence language.

### PhiKernel Hooks

- Guard callouts before command/tool execution.
- Runtime anomaly feed into posture updates.
- Policy enforcement checkpoints around privileged actions.

### PhiOS View Model

- Operator posture dashboard (SAFE → QUARANTINED).
- Incident timeline and continuity deltas.
- Human review pathways for quarantine/seal events.

## 10) Proposed Repo Structure

Future-facing structure preserving ANC economic lineage:

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

## 11) Recommended Detector Set for v1.0

Initial detector families:

- Jailbreak detector.
- Prompt injection detector.
- Suspicious command/code detector.
- Memory contamination detector.
- Output leakage detector.
- Drift/anomaly detector.

These should be introduced incrementally with explicit false-positive/false-negative evaluation.

## 12) Build Phases

### Phase 1: Core Trust Engine

- Core models (signals, posture, decisions, incidents).
- Scoring and posture-transition rules.
- Policy action selection framework.

### Phase 2: Detectors

- Implement baseline detector set.
- Add calibration harness and threshold tuning.

### Phase 3: PhiKernel Hooks

- Integrate guard checkpoints with runtime and tools.
- Enforce action pathways (`sandbox`, `quarantine`, `seal`).

### Phase 4: PhiOS Surface

- Posture visualization.
- Incident observability and operator workflows.

## 13) Success Criteria

ANC v1.0 is successful when:

- Economic simulator lineage remains intact and reproducible.
- Threat signals can be normalized into stable posture transitions.
- Guard decisions are deterministic, auditable, and policy-traceable.
- Quarantine and recovery flows maintain continuity integrity.
- Integration hooks support sovereign, local-first operation without cloud dependency assumptions.

## 14) Immediate Next Step

Implement the **core models + posture + scoring engine** first:

1. Define `ThreatSignal`, `CoherenceSecurityState`, `GuardDecision`, and `IncidentRecord` as concrete Python models.
2. Implement baseline scoring for \(\Theta(t)\), \(\Lambda\), \(A_{int}\), \(R_v\), and derived \(C_{sec}\).
3. Add a deterministic posture-transition state machine with tests.

This creates a stable foundation for detectors and runtime integration in later phases.
