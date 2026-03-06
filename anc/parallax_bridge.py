"""Optional ANC bridge to the wider Parallax stack.

All methods degrade safely when optional dependencies are missing.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable

from anc import __tiekat_version__


class ParallaxBridge:
    bridge_version = "0.2.0"

    def __init__(self) -> None:
        self._phios = self._try_import("phios")
        self._tbrc = self._try_import("tbrc")

    @staticmethod
    def _try_import(module_name: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "phios_available": self._phios is not None,
            "tbrc_available": self._tbrc is not None,
            "tiekat_version": __tiekat_version__,
            "bridge_version": self.bridge_version,
        }

    @staticmethod
    def _empty_schema(epoch: int, permit: bool) -> Dict[str, Any]:
        return {
            "network_lt": 0.0,
            "top_validator": None,
            "top_lt": 0.0,
            "bottom_validator": None,
            "bottom_lt": 0.0,
            "sovereign_count": 0,
            "total_validators": 0,
            "permit": bool(permit),
            "epoch": int(epoch),
            "tiekat_version": __tiekat_version__,
        }

    def export_network_lt(self, validators: Iterable[Any], epoch: int, permit: bool) -> Dict[str, Any]:
        try:
            vals = list(validators or [])
            if not vals:
                return self._empty_schema(epoch, permit)

            def _vid(v: Any) -> Any:
                return getattr(v, "vid", None)

            def _lt(v: Any) -> float:
                return float(getattr(v, "lt_score", 0.0) or 0.0)

            scored = sorted(vals, key=_lt, reverse=True)
            top = scored[0]
            bottom = scored[-1]
            lts = [_lt(v) for v in vals]
            sovereign_count = sum(1 for lt in lts if lt >= 0.7)
            return {
                "network_lt": sum(lts) / len(lts) if lts else 0.0,
                "top_validator": _vid(top),
                "top_lt": _lt(top),
                "bottom_validator": _vid(bottom),
                "bottom_lt": _lt(bottom),
                "sovereign_count": sovereign_count,
                "total_validators": len(vals),
                "permit": bool(permit),
                "epoch": int(epoch),
                "tiekat_version": __tiekat_version__,
            }
        except Exception:
            return self._empty_schema(epoch, permit)

    def store_epoch_in_tbrc(self, payload: Dict[str, Any]) -> None:
        if self._tbrc is None:
            return
        try:
            hook = getattr(self._tbrc, "store_epoch", None)
            if callable(hook):
                hook(payload)
        except Exception:
            return

    def store_simulation_run(self, payload: Dict[str, Any]) -> None:
        if self._tbrc is None:
            return
        try:
            hook = getattr(self._tbrc, "store_run", None)
            if callable(hook):
                hook(payload)
        except Exception:
            return

    def notify_resonance_moment(self, epoch: int) -> None:
        if epoch % 369 != 0 or self._phios is None:
            return
        try:
            hook = getattr(self._phios, "notify_resonance", None)
            if callable(hook):
                hook(epoch)
        except Exception:
            return
