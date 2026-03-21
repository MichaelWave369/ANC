"""Reporting utilities for ANC simulation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from anc.tiekat_v81 import sparkline


class ANCReportGenerator:
    def _version_tag(self, summary: Dict[str, Any]) -> str:
        return str(summary.get("version", "0.2.0"))

    def _v03_blocks(self, summary: Dict[str, Any]) -> str:
        if "continuity_windows" not in summary:
            return ""
        return f"""
## v0.3 Continuity Diagnostics
- Continuity windows: `{summary.get('continuity_windows', 0)}`
- Final continuity score: `{summary.get('continuity_score', 0.0):.6f}`
- Recovery events: `{summary.get('recovery_events', 0)}`
- Training baseline: `{summary.get('training_baseline_score', 0.0):.6f}`
- Training delta: `{summary.get('training_delta', 0.0):.6f}`

## v0.3 Branch + Memory Outputs
- Branch selection CSV: `{{run_id}}_branches.csv`
- Continuity CSV: `{{run_id}}_continuity.csv`
- Memory crystal vault: `{summary.get('vault_file', 'n/a')}`
"""

    def generate_report(
        self,
        summary: Dict[str, Any],
        metrics_rows: List[Dict[str, Any]],
        output_dir: str = "out",
    ) -> str:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        run_id = summary.get("run_id", "anc_sim")
        version_tag = self._version_tag(summary)
        lt_values = [float(r.get("mean_lt", 0.0)) for r in metrics_rows]
        psi_values = [float(r.get("mean_psi_b", 0.0)) for r in metrics_rows]

        chart_file = out_path / f"{run_id}_lt_chart.txt"
        chart_text = (
            f"ANC v{version_tag} L(t) / Ψᵇ trend\n"
            f"L(t):  {sparkline(lt_values, width=64)}\n"
            f"Psi_b: {sparkline(psi_values, width=64)}\n"
        )
        chart_file.write_text(chart_text, encoding="utf-8")

        v03 = self._v03_blocks(summary).replace("{run_id}", run_id)
        report_file = out_path / f"{run_id}_report.md"
        md = f"""# ANC v{version_tag} Simulation Report

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

## Active Features
- TIEKAT version: `{summary.get('tiekat_version', '8.1')}`
- Parallax bridge export: `True`

## Resonance Events
- 369 resonance events: `{summary.get('resonance_events', [])}`
{v03}
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
        version_tag = self._version_tag(summary)
        lt_values = [float(r.get("mean_lt", 0.0)) for r in metrics_rows]
        psi_values = [float(r.get("mean_psi_b", 0.0)) for r in metrics_rows]
        v03_line = ""
        if "continuity_windows" in summary:
            v03_line = (
                f"\n- Continuity score: `{summary.get('continuity_score', 0.0):.6f}`"
                f"\n- Recovery events: `{summary.get('recovery_events', 0)}`"
                f"\n- Training delta: `{summary.get('training_delta', 0.0):.6f}`"
            )

        file_path = out_path / f"{run_id}_gabriel.md"
        text = f"""# ANC v{version_tag} Technical Summary (for Gabriel Cardona)

- TIEKAT version: `{summary.get('tiekat_version', '8.1')}`
- Epochs: `{summary.get('epochs', 0)}`
- Final mean L(t): `{summary.get('final_mean_lt', 0.0):.6f}`
- Final mean Psi_b: `{summary.get('final_mean_psi_b', 0.0):.6f}`
- Permit rate: `{summary.get('permit_rate', 0.0):.6f}`
- Nakamoto(33%): `{summary.get('nakamoto_33', 0)}`
- Gini: `{summary.get('gini_stake', 0.0):.6f}`{v03_line}

Trend snapshots:
- L(t): `{sparkline(lt_values, width=56)}`
- Psi_b: `{sparkline(psi_values, width=56)}`
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
