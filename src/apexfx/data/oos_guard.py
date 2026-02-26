"""Out-of-Sample Guard — protects the sacred final test set from contamination.

The OOS set is the LAST line of defence against overfitting to history.
It must never be touched during:
  - Model training (any stage)
  - Walk-forward / cross-validation
  - Hyperparameter optimisation (Optuna)
  - Feature selection or engineering iterations

Only unlock it *once* with :meth:`OOSGuard.unlock_oos` when the model is
fully frozen and you are ready for the final, irreversible production evaluation.

Usage
-----
    guard = OOSGuard(data, oos_fraction=0.2, data_dir="./data")
    train_pool, oos = guard.split()          # oos is SEALED — do not touch
    trainer = Trainer(config, real_data=train_pool)
    trainer.train()

    # --- months later, model in production ---
    final_oos = guard.unlock_oos()           # logs irreversible access
    results = evaluate(model, final_oos)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

_MANIFEST_FILENAME = "oos_manifest.json"


class OOSGuard:
    """Enforces a strict separation between the training pool and the OOS set.

    Parameters
    ----------
    data:
        Full historical dataset (chronologically sorted).
    oos_fraction:
        Fraction of *total* data reserved as OOS (default 0.2 = last 20%).
    data_dir:
        Root data directory where the OOS manifest is persisted.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        oos_fraction: float = 0.2,
        data_dir: str | Path = "./data",
    ) -> None:
        if not (0.0 < oos_fraction < 1.0):
            raise ValueError(f"oos_fraction must be in (0, 1), got {oos_fraction}")

        self._data = data.reset_index(drop=True)
        self._oos_fraction = oos_fraction
        self._data_dir = Path(data_dir)
        self._manifest_path = self._data_dir / _MANIFEST_FILENAME

        n = len(self._data)
        self._split_idx = int(n * (1.0 - oos_fraction))

        self._log_creation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self) -> tuple[pd.DataFrame, None]:
        """Return the training pool.  The OOS set is intentionally withheld.

        Returns
        -------
        train_pool : pd.DataFrame
            Data available for training and walk-forward validation.
        oos_placeholder : None
            The OOS set is NOT returned here.  Use :meth:`unlock_oos` only
            when the model is fully finalised and production-ready.
        """
        train_pool = self._data.iloc[: self._split_idx].reset_index(drop=True)

        n_total = len(self._data)
        n_train = len(train_pool)
        n_oos = n_total - n_train

        logger.info(
            "OOSGuard split",
            total_bars=n_total,
            train_bars=n_train,
            oos_bars=n_oos,
            oos_fraction=self._oos_fraction,
            oos_start_idx=self._split_idx,
            oos_start_time=str(self._data.iloc[self._split_idx].get("time", "unknown")),
        )

        return train_pool, None  # OOS is intentionally withheld

    def unlock_oos(self, reason: str = "final evaluation") -> pd.DataFrame:
        """Access the OOS set.  This action is permanent and logged.

        .. warning::
            **IRREVERSIBLE.**  Once you look at the OOS set your model is no
            longer truly out-of-sample.  Call this function *only once*, after
            all development is complete and the model is frozen.

        Parameters
        ----------
        reason:
            Free-text description of why the OOS set is being accessed.
            Stored in the manifest for audit purposes.

        Returns
        -------
        pd.DataFrame
            The held-out OOS portion of the dataset.
        """
        oos_data = self._data.iloc[self._split_idx :].reset_index(drop=True)
        self._record_unlock(reason, n_bars=len(oos_data))

        logger.warning(
            "OOS SET UNLOCKED — this data must not influence future model changes",
            reason=reason,
            oos_bars=len(oos_data),
            oos_fraction=self._oos_fraction,
            split_idx=self._split_idx,
        )

        return oos_data

    @property
    def split_index(self) -> int:
        """Bar index at which the OOS set begins."""
        return self._split_idx

    @property
    def oos_fraction(self) -> float:
        return self._oos_fraction

    @property
    def n_train_bars(self) -> int:
        return self._split_idx

    @property
    def n_oos_bars(self) -> int:
        return len(self._data) - self._split_idx

    # ------------------------------------------------------------------
    # Manifest helpers (audit trail)
    # ------------------------------------------------------------------

    def _log_creation(self) -> None:
        manifest = self._load_manifest()
        entry = {
            "event": "guard_created",
            "timestamp": _utcnow(),
            "total_bars": len(self._data),
            "split_idx": self._split_idx,
            "oos_fraction": self._oos_fraction,
            "n_train_bars": self._split_idx,
            "n_oos_bars": len(self._data) - self._split_idx,
        }
        manifest["history"].append(entry)
        self._save_manifest(manifest)

    def _record_unlock(self, reason: str, n_bars: int) -> None:
        manifest = self._load_manifest()
        entry = {
            "event": "oos_unlocked",
            "timestamp": _utcnow(),
            "reason": reason,
            "n_bars": n_bars,
            "split_idx": self._split_idx,
        }
        manifest["history"].append(entry)
        manifest["unlock_count"] = manifest.get("unlock_count", 0) + 1
        self._save_manifest(manifest)

        if manifest["unlock_count"] > 1:
            logger.error(
                "OOS set has been unlocked multiple times — results are no longer "
                "truly out-of-sample!",
                unlock_count=manifest["unlock_count"],
            )

    def _load_manifest(self) -> dict:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {"history": [], "unlock_count": 0}

    def _save_manifest(self, manifest: dict) -> None:
        with open(self._manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
