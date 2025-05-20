import pandas as pd
import numpy as np
import logging


def flag_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_col_name = f"{col}_outlier"
        df[outlier_col_name] = df[col].apply(
            lambda x: "LOW" if x < lower_bound else (
                "HIGH" if x > upper_bound else "NORMAL")
        )

    return df


def compute_line_level_rate_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes:
    - rate_ratio_normal: Normalized comparison of modelled vs. actual unit rate
    - pct_difference: Percentage savings or overspend at the invoice level
    - savings_flag: "SAVINGS" if model is cheaper, "LOSS" if model is more expensive
    """
    required_cols = {"realistic_optimal_cost",
                     "freight_per_invoice", "invoice_commodity_quantity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required_cols - set(df.columns)}")

    # Rate ratio (normalized per unit)
    df["rate_ratio_normal"] = df.apply(
        lambda row: (row["realistic_optimal_cost"] / row["invoice_commodity_quantity"]) /
                    (row["freight_per_invoice"] /
                     row["invoice_commodity_quantity"])
        if all(pd.notnull([row["realistic_optimal_cost"], row["freight_per_invoice"], row["invoice_commodity_quantity"]])
               ) and row["freight_per_invoice"] != 0 and row["invoice_commodity_quantity"] != 0
        else None,
        axis=1
    )

    # % Difference (actual - modelled) / actual
    df["pct_difference"] = df.apply(
        lambda row: ((row["freight_per_invoice"] -
                     row["realistic_optimal_cost"]) / row["freight_per_invoice"])
        if pd.notnull(row["realistic_optimal_cost"]) and pd.notnull(row["freight_per_invoice"]) and row["freight_per_invoice"] != 0
        else None,
        axis=1
    )

    # Flag: "SAVINGS" if model is cheaper, "LOSS" if model is more expensive
    df["savings_flag"] = df["pct_difference"].apply(
        lambda x: "SAVINGS" if x is not None and x > 0 else (
            "LOSS" if x is not None and x < 0 else None)
    )

    return df


def compute_site_level_freight_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes site-level freight ratio:
    - freight_ratio_normal_site = sum(realistic_optimal_cost) / sum(freight_per_invoice)
    Adds this ratio back to each row based on its site.
    """
    if not {"realistic_optimal_cost", "freight_per_invoice", "site"}.issubset(df.columns):
        raise ValueError(
            "Missing required columns for freight ratio calculation.")

    # Group by site and calculate the ratio
    ratio_df = df.groupby("site").agg({
        "realistic_optimal_cost": "sum",
        "freight_per_invoice": "sum"
    }).reset_index()

    ratio_df["freight_ratio_normal_site"] = ratio_df.apply(
        lambda row: row["realistic_optimal_cost"] / row["freight_per_invoice"]
        if row["freight_per_invoice"] != 0 else None,
        axis=1
    )

    # Merge back to the main dataframe
    df = df.merge(
        ratio_df[["site", "freight_ratio_normal_site"]], on="site", how="left")

    return df
