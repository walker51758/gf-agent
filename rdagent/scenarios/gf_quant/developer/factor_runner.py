"""Qlib runner with the GF scenario's stricter de-duplication policy."""

from __future__ import annotations

import pandas as pd

from rdagent.core.exception import FactorEmptyError
from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.scenarios.gf_quant.dedup import FactorDeduplicator, FactorSignature
from rdagent.scenarios.qlib.developer.factor_runner import QlibFactorRunner


class GFFactorRunner(QlibFactorRunner):
    """Drop candidates similar by absolute Pearson or Spearman correlation."""

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        deduplicator = FactorDeduplicator(
            threshold=GF_SCENARIO_SETTINGS.dedup_threshold,
            min_overlap=GF_SCENARIO_SETTINGS.dedup_min_overlap,
        )
        existing = [
            FactorSignature(str(column), str(column), values=SOTA_feature[column].dropna())
            for column in SOTA_feature.columns
        ]
        keep: list[object] = []
        for column in new_feature.columns:
            signature = FactorSignature(str(column), str(column), values=new_feature[column].dropna())
            if not deduplicator.compare(signature, existing).is_duplicate:
                keep.append(column)
                existing.append(signature)
        if not keep:
            raise FactorEmptyError(
                "All new factors duplicate the library under the configured GF Pearson/Spearman threshold."
            )
        return new_feature.loc[:, keep]
