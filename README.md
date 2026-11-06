# ANC | Anchor

**Coherence-aware risk signals for sovereign software.**  
A PHI369 Labs research and integration project.

ANC began as a coherence-weighted cryptocurrency **simulation**. Its active engineering direction is a **local-first, inspectable trust-advisory layer** for the PhiOS ecosystem: detect selected risky patterns, summarize system posture, and pass bounded evidence to an independently authorized policy engine.

> **Project status:** Experimental components and testable interfaces, **not** a production security product, deployed cryptocurrency, operating blockchain, or live authorization service.

## What exists today

| Component | Implemented | Important boundary |
| --- | --- | --- |
| Threat signal detectors | Deterministic rules for selected prompt-injection, jailbreak, runtime-command, memory-contamination, and output-leakage patterns | Pattern matches are signals, not comprehensive protection or proof of malicious intent |
| Coherence trust-state model | 12-face security field, topology summaries, posture scoring, and recommended responses | Model outputs are research heuristics, not authenticated identity or validated security guarantees |
| PhiKernel-style adapters | Pre/post-service, memory-write, and output-commit guard interpretations | Produce in-process plans and outcomes; do **not** launch sandboxes, terminate processes, or enforce OS permissions |
| ANC ↔ PhiShield advisory R1 | Strict, request-bound, expiring JSON-compatible risk envelope with negative-path tests | Unsigned and advisory-only; cannot issue tokens, approve capabilities, or authenticate a remote producer |
| PhiOS observability model | Structures for posture, affected faces, incident views, and recommended actions | Presentation data, not a deployed security dashboard |
| Anchor v0.1–v0.3 simulations | Preserved validator, anti-capture, network-shock, and continuity experiments | Simulated economics and consensus; **no** live ANC network, mainnet, wallet, or payment requirement |

Key source: [core models](anc/core/models.py) · [detectors](anc/detectors) · [integration adapters](anc/integrations) · [simulators](anchor_sim_v0_3.py)

## Where ANC fits in the Phi ecosystem

```text
PhiOS / PhiCloud / apps / operators
                  |
            request + context
                  |
          ANC: detect & assess
          risk / posture / signals
                  |
           bounded advisory
                  |
         PhiShield policy layer
        (reference consumer in
              PhiCloud)
                  |
     independently verified identity,
       capabilities, and consent
                  |
       PhiOS / service authority
         makes final decision
```

**Separation of duties matters:**

- **PhiID** is the identity direction. A coherence score does not prove who somebody is.
- **ANC** assesses selected risk signals and continuity. Its results **never** grant authority.
- **PhiCloud's PhiShield reference consumer** evaluates ANC advisories against separately supplied authority and consent context; it returns only `DENY`, `REVIEW`, or `CANDIDATE`. A candidate is **not** execution permission.
- **PhiOS** retains independent, explicit authorization for local files, devices, code execution, and other effects. Offline operation must not require ANC coins, accounts, or cloud services.

The PhiCloud-side consumer and its cross-repository smoke tests were introduced in [PhiCloud PR #4](https://github.com/MichaelWave369/PhiCloud/pull/4). It is a tested **local reference implementation**, not a network-exposed PhiShield deployment.

## Try the code

Requirements: **Python 3.11+**. From the repository root:

```bash
git clone https://github.com/MichaelWave369/ANC.git
cd ANC

# Run the ANC / PhiShield advisory integration tests
python -m unittest -v tests.test_phishield_contract

# Run the preserved v0.3 economic/continuity simulator
python anchor_sim_v0_3.py
```

The simulator uses its configured run parameters and writes research output under `out/`. Earlier experiments remain available as `anchor_sim_v0_2.py` and `anchor_sim_v0_1_1a.py`.

To explore the guard interface *without executing a protected action*:

```python
from anc.integrations.phikernel_soc_adapter import guard_input_text
from anc.integrations.phishield_contract import make_anc_advisory

assessment = guard_input_text(
    "Summarize the current service state",
    source_ref="local-example",
)
advisory = make_anc_advisory(
    assessment,
    request_id="example-369",
    scope="pre_service",
)
print(advisory.to_payload())
# A risk advisory is not an authentication result or permission to execute.
```

Some other test modules use `pytest`; install the appropriate test tooling before running the entire test suite. The [GitHub Actions contract workflow](.github/workflows/anc-phishield-contract.yml) covers the advisory integration across multiple Python versions.

## Current limitations

ANC does **not** yet provide:

- Production-ready malware detection, proof that prompts are safe, or a substitute for code review and sandboxing.
- Signed risk attestations, authenticated remote transport, replay-resistant service channels, or trusted execution enforcement.
- Real PhiID token verification, live service authorization, a deployed PhiShield gateway, or a hosted security observatory.
- A functioning cryptocurrency network, tradable token, mainnet, or network-wide consensus finality.

A clean score is **not** proof of safety. A flagged phrase may be benign. Real deployments require threat modeling, adversarial testing, false-positive evaluation, independently verified identity and consent, and security review.

## Engineering direction

1. **Harden the advisory boundary:** authenticated provenance, nonce/replay protection, expiry, schema compatibility, and negative tests.
2. **Integrate with policy safely:** real PhiShield verification and independently governed authority; keep ANC limited to evidence and risk interpretation.
3. **Measure reliability:** false positives/negatives, detector bypasses, tampering, unavailable signals, and offline failure modes.
4. **Improve visibility:** incident receipts, explainable posture transitions, and operator review without silently granting privileges.
5. **Preserve economic research separately:** maintain reproducibility of validator simulations without tying accounts or everyday security to coins.

These are **plans**, not claims of completed security capabilities.

## Documentation and project history

- [ANC ↔ PhiShield R1 contract](docs/ANC_PHISHIELD_R1_CONTRACT.md) — concrete schema, examples, boundaries, and test coverage.
- [Trust-plane architecture (TIEKAT v69-aligned)](docs/anc_v1_architecture.md) — proposed topology model and further work.
- [Changelog](CHANGELOG.md) — earlier validator and continuity simulation milestones.
- [Historical concepts and artwork](docs/PROJECT_HISTORY.md) — original Anchor/CWPoR positioning, the former Parallax-era materials, and archived image links.
- [License history and rights review](LICENSE_HISTORY.md) — prior GPL/AGPL releases and MIT transition.

### Research provenance

ANC's TIEKAT/HQRMA mathematical lineage acknowledges **Hemavit**, as credited in the original research. PHI369 Labs and Michael Hughes developed the ANC project and associated integration work. Attribution is maintained without asserting ownership of independently supplied material or production validation of research hypotheses.

## License

The repository's original software is published under the [MIT License](LICENSE) as of November 6, 2026, to the extent the rights holders can grant MIT permissions. Historical GPL-3.0 and AGPL-3.0 grants remain in effect for prior releases. Independent third-party content, including the whitepaper and infographics, must be reviewed separately: the repository's root license does **not** automatically relicense those assets.

**PHI369 Labs · People Above Profit · Ledger Above Ego**
