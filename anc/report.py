"""Reporting utilities for ANC v0.2 simulation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from anc.tiekat_v81 import sparkline


class ANCReportGenerator:
    def generate_report(
        self,
        summary: Dict[str, Any],
        metrics_rows: List[Dict[str, Any]],
        output_dir: str = "out",
    ) -> str:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        run_id = summary.get("run_id", "anc_sim")
        lt_values = [float(r.get("mean_lt", 0.0)) for r in metrics_rows]
        psi_values = [float(r.get("mean_psi_b", 0.0)) for r in metrics_rows]

        chart_file = out_path / f"{run_id}_lt_chart.txt"
        chart_text = (
            "ANC v0.2 L(t) / Ψᵇ trend\n"
            f"L(t):  {sparkline(lt_values, width=64)}\n"
            f"Psi_b: {sparkline(psi_values, width=64)}\n"
        )
        chart_file.write_text(chart_text, encoding="utf-8")

        report_file = out_path / f"{run_id}_report.md"
        md = f"""# ANC v0.2 Simulation Report

## Run Metadata
- Run ID: `{run_id}`
- Epochs: `{summary.get('epochs', 0)}`
- Validators: `{summary.get('n_validators', 0)}`
- Delegators: `{summary.get('n_delegators', 0)}`
- Seed: `{summary.get('seed', 369369)}`

## Final Metrics
- Final mean L(t): `{summary.get('final_mean_lt', 0.0):.6f}`
- Final mean Psi_b: `{summary.get('final_mean_psi_b', 0.0):.6f}`
- Permit rate: `{summary.get('permit_rate', 0.0):.6f}`

## Trends
- L(t): `{sparkline(lt_values, width=64)}`
- Psi_b: `{sparkline(psi_values, width=64)}`

## Active v8.1 Features
- Hemavit path integral: `{summary.get('use_hemavit_issuance', False)}`
- HQRMA smoothing: `{summary.get('use_hqrma_smoothing', False)}`
- 12-axis telemetry: `{summary.get('use_12axis_telemetry', False)}`
- Parallax bridge export: `True`

## Resonance Events
- 369 resonance events: `{summary.get('resonance_events', [])}`

ANC shares the same unified **L(t)** formula with PhiOS and TBRC.
"""
        report_file.write_text(md, encoding="utf-8")
        return str(report_file)

    def generate_gabriel_summary(
        self,
        summary: Dict[str, Any],
        metrics_rows: List[Dict[str, Any]],
        output_dir: str = "out",
    ) -> str:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        run_id = summary.get("run_id", "anc_sim")
        lt_values = [float(r.get("mean_lt", 0.0)) for r in metrics_rows]
        psi_values = [float(r.get("mean_psi_b", 0.0)) for r in metrics_rows]

        file_path = out_path / f"{run_id}_gabriel.md"
        text = f"""# ANC v0.2 Technical Summary (for Gabriel Cardona)

ANC v0.2 migrates coherence math from TIEKAT v6.6 to v8.1 while preserving additive compatibility.

- TIEKAT version: `{summary.get('tiekat_version', '8.1')}`
- Epochs: `{summary.get('epochs', 0)}`
- Final mean L(t): `{summary.get('final_mean_lt', 0.0):.6f}`
- Final mean Psi_b: `{summary.get('final_mean_psi_b', 0.0):.6f}`
- Permit rate: `{summary.get('permit_rate', 0.0):.6f}`
- Nakamoto(33%): `{summary.get('nakamoto_33', 0)}`
- Gini: `{summary.get('gini_stake', 0.0):.6f}`

Trend snapshots:
- L(t): `{sparkline(lt_values, width=56)}`
- Psi_b: `{sparkline(psi_values, width=56)}`

v8.1 runtime pathways enabled:
- 12-axis telemetry vector
- HQRMA renormalization flow
- Hemavit path-integral issuance gate
- Network bridge export for Parallax interoperability
"""
        file_path.write_text(text, encoding="utf-8")
        return str(file_path)

    def generate_bridge_export(
        self,
        network_lt: Dict[str, Any],
        output_dir: str = "out",
        run_id: str = "anc_sim",
    ) -> str:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        file_path = out_path / f"{run_id}_bridge.json"
        file_path.write_text(json.dumps(network_lt, indent=2, sort_keys=True), encoding="utf-8")
        return str(file_path)
