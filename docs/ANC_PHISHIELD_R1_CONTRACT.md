# ANC ↔ PhiShield R1: Advisory Trust Contract

**Status:** Implemented as a local, side-effect-free ANC adapter, reference consumer, and automated tests. **Not** a deployed PhiShield service, authenticated network connection, permission system, or audited security control.

## Why

ANC already detects some input, runtime, memory, and output risk signals and calculates a topology-aware trust posture. PhiCloud names **PhiShield** as the eventual policy-enforcement boundary. As of this milestone, PhiShield has **no separate repository or live service** confirmed in the connected account.

The goal is a narrow handoff that lets PhiShield consume ANC *risk evidence* without mistaking ANC for an identity provider, an authority source, or an enforcement mechanism.

```text
request + trusted server-side identity / action context
  ├── ANC detector -> scoring -> advisory (read-only risk evidence)
  └── PhiID / PhiOS / PhiShield independent authority and consent checks
              |
              v
     PhiShield policy decision (future)
              |
              v
   PhiOS / PhiKernel performs separately authorized action (future)
```

**Invariant:** A high ANC coherence score never repairs an invalid signature, grants a capability, overrides consent, or makes an otherwise forbidden action lawful.

## Wire schema

`phi369.anc.phishield.advisory.v1` is strict and versioned.

| Field | Meaning |
| --- | --- |
| `schema` | Exact version string |
| `request_id` | Public correlation ID for *one* action, not a secret |
| `scope` | `pre_service`, `post_service`, `memory_write`, `output_commit` |
| `posture` | ANC posture: `safe`, `watch`, `degraded`, `hostile`, `quarantined` |
| `recommended_action` | ANC recommendation, not an authorized operation |
| `signal_count` | Number of detected signals; does not carry raw evidence |
| `weakest_face`, `weakest_face_score` | Topology summary, not a cryptographic proof |
| `anchor_integrity` | ANC model metric, **not** signed ledger finality |
| `observed_at`, `not_after` | Timezone-aware timestamps; lifetime no more than 120 seconds |
| `authority_granted` | Must always be exactly `false` |

Unknown fields, missing fields, invalid types/ranges, unknown versions, mismatched request IDs or scopes, future-dated envelopes, and expired envelopes are rejected. No user identifiers, prompts, memory contents, tokens, addresses, credentials, or raw detector excerpts are forwarded.

**Security caveat:** Strict parsing is not proof of origin. This R1 payload is **unsigned** and can be forged by an attacker able to cross the transport boundary. Only pass data across a trusted in-process boundary in this milestone. A real remote adapter needs authenticated transport, producer identity, replay controls, and policy-controlled trust roots.

## Sample Python flow

```python
from datetime import datetime, timezone
from anc.integrations.phikernel_soc_adapter import guard_input_text
from anc.integrations.phishield_contract import (
    make_anc_advisory,
    reference_phishield_gate,
)

now = datetime.now(timezone.utc)
guard = guard_input_text("Summarize these metrics", source_ref="local-test")
advisory = make_anc_advisory(
    guard, request_id="request-369", scope="pre_service", now=now
)

# Reference gate ONLY: the flags below must originate from a separate
# trusted policy engine, never the client or the ANC result.
outcome = reference_phishield_gate(
    advisory.to_payload(),
    request_id="request-369",
    scope="pre_service",
    authority_verified=False,
    consent_verified=False,
    now=now,
)
assert outcome.disposition == "deny"
assert not outcome.authoritative
```

### Inert reference dispositions

- `deny`: bad/missing/stale evidence, no independently verified authority/consent, hostile risk posture, or blocking ANC recommendation.
- `review`: moderate/uncertain risk evidence requires an operator.
- `candidate`: clean ANC signal **only**. Still not authorization to execute, write, publish, or transmit.

The reference gate deliberately has **no executable output**, credential management, network listener, file-system access, or process-control hook. The caller is responsible for independent action authorization, capability scope, consent, sandbox selection, and final enforcement.

## Threat cases exercised

The tests pass a live ANC detector/scorer result through a JSON serialization boundary to the independent reference consumer, and check:

1. Clean evidence never grants authority, even when a risk candidate exists.
2. Denial without verified capability/consent.
3. Prompt-injection detection.
4. Suspicious shell pipeline detection.
5. Memory-contamination risk and scope separation.
6. Request mismatch and replay-style misuse.
7. Expired, future-dated, and excessive lifetime envelopes.
8. Forged `authority_granted`, unknown schema, injected extra fields, malformed scores.
9. Sensitive source text never appears in the advisory payload.
10. Missing/invalid advisory fails closed in the reference consumer.

Run with Python 3.11+:

```bash
python -m unittest -v tests.test_phishield_contract
```

A dedicated GitHub Actions workflow runs the same suite for pull requests and main-branch pushes.

## Next integration milestone (requires a real PhiShield service)

- Define a separately owned PhiShield policy interface and canonical action/resource/subject model.
- Establish authenticated local IPC with issuer identification, replay-resistant nonces, and request binding; later add mutual TLS if remote.
- Use a mature OIDC/WebAuthn stack for PhiID, with verifier-side audience, issuer, expiration, and signature checks. Do not use ANC scores to validate credentials.
- Make PhiOS/PhiKernel enforcement an independently tested, deny-by-default operation with explicit operator approval for consequential actions.
- Measure false positives/negatives, adversarial bypasses, timeouts, degraded/offline handling, and operator recovery. Add signed audit receipts without putting sensitive payloads on a public chain.

ANC economic simulations, staking, and coin ownership have **no role in access control**. The trust subsystem must work offline and without ANC tokens.

This is a versioned interface milestone, not a claim of production security.
