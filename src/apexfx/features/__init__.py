"""Feature engineering pipeline for ApexFX Quantum."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors."""

    @abstractmethod
    def extract(
        self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Extract features from bar data. Returns DataFrame with same index, new columns."""
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature column names produced by this extractor."""
        ...
