from __future__ import annotations

import numpy as np
import pandas as pd


QUERY_CONTEXT_COLUMNS = ("sig1", "sig2", "sig7", "sig8")
LOG_FEATURE_COLUMNS = ("sig3", "sig4", "sig5", "sig6")


def add_engineered_features(
    df: pd.DataFrame,
    *,
    include_query_context: bool = True,
    include_query_aggregates: bool = False,
) -> pd.DataFrame:
    """Apply the feature engineering used by the active project scripts."""
    df = df.copy()

    for column in LOG_FEATURE_COLUMNS:
        if column in df.columns:
            df[f"log_{column}"] = np.log1p(df[column].astype(float))

    eps = 1e-6
    if {"sig1", "sig2"}.issubset(df.columns):
        df["sig_ratio_21"] = df["sig2"] / (df["sig1"] + eps)
    if {"sig1", "sig7", "sig8"}.issubset(df.columns):
        df["sig_sum_178"] = df["sig1"] + df["sig7"] + df["sig8"]
    if {"is_homepage", "sig2"}.issubset(df.columns):
        df["hp_sig2"] = df["is_homepage"] * df["sig2"]

    if include_query_context and "query_id" in df.columns:
        grouped = df.groupby("query_id", group_keys=False)
        for column in QUERY_CONTEXT_COLUMNS:
            if column not in df.columns:
                continue

            df[f"{column}_qrank"] = grouped[column].rank(pct=True)
            mean = grouped[column].transform("mean")
            std = grouped[column].transform("std").replace(0, np.nan)
            df[f"{column}_qz"] = ((df[column] - mean) / std).fillna(0.0)

            if include_query_aggregates:
                df[f"{column}_qmean"] = mean
                df[f"{column}_qmax"] = grouped[column].transform("max")
                df[f"{column}_qmin"] = grouped[column].transform("min")

    return df
