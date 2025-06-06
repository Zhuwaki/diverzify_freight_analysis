import pandas as pd
import numpy as np
import logging


def flag_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Flags outliers in specified numeric columns using IQR method.
    Adds a column per input with suffix '_outlier' containing:
    - 'LOW', 'HIGH', or 'OK'
    """
    df = df.copy()
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_col_name = f"{col}_outlier"

        df[outlier_col_name] = df[col].apply(
            lambda x: (
                "LOW" if pd.notna(x) and x < lower_bound else
                "HIGH" if pd.notna(x) and x > upper_bound else
                "OK" if pd.notna(x) else "MISSING"
            )
        )

    return df


def compute_line_level_rate_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the line-level rate ratio by normalizing both modelled and actual costs
    by quantity to compare per-unit rates. Avoids division by zero.
    """
    df = df.copy()

    required_cols = {
        "realistic_optimal_cost",
        "freight_per_invoice",
        "invoice_commodity_quantity"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Avoid divide-by-zero: set invoice_commodity_quantity = NaN where it's zero or null
    df["invoice_commodity_quantity"] = pd.to_numeric(
        df["invoice_commodity_quantity"], errors="coerce")
    df.loc[df["invoice_commodity_quantity"] ==
           0, "invoice_commodity_quantity"] = pd.NA

    # Compute per-unit rates
    df["xgs_rate"] = pd.to_numeric(
        df["realistic_optimal_cost"], errors="coerce") / df["invoice_commodity_quantity"]
    df["historical_rate"] = pd.to_numeric(
        df["freight_per_invoice"], errors="coerce") / df["invoice_commodity_quantity"]

    # Replace 0s in historical_rate with NaN in-place to avoid unsafe division
    df.loc[df["historical_rate"] == 0, "historical_rate"] = pd.NA

    # Compute safely
    df["rate_ratio_normal"] = df["xgs_rate"] / df["historical_rate"]
    df["pct_difference"] = (
        df["xgs_rate"] - df["historical_rate"]) / df["historical_rate"] * 100

    # Flag savings
    df["savings_flag"] = df["pct_difference"].apply(
        lambda x: "SAVINGS" if pd.notna(x) and x > 0 else "LOSS"
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


def append_group_stats_to_df(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """
    Computes mean, median, mode(s), and average for specified columns grouped by
    ['site', 'new_commodity_group'] and appends them back to the original DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'site' and 'new_commodity_group'
        value_columns (list): List of numeric columns to summarize

    Returns:
        pd.DataFrame: Original DataFrame with additional columns per stat
    """
    df = df.copy()

    # Group and compute stats
    stats = df.groupby(['site', 'new_commodity_group'])[value_columns].agg(
        ['mean', 'median', lambda x: x.mode().tolist(), 'sum', 'count']
    )

    # Rename lambda-generated column for mode
    stats.columns = ['_'.join(
        [col[0], col[1] if col[1] != '<lambda>' else 'mode']) for col in stats.columns]

    # Compute average = sum / count
    for col in value_columns:
        stats[f"{col}_average"] = stats[f"{col}_sum"] / stats[f"{col}_count"]

    # Drop raw sum and count if not needed
    stats = stats.drop(columns=[
                       f"{col}_sum" for col in value_columns] + [f"{col}_count" for col in value_columns])

    # Reset index and merge back to original DataFrame
    stats = stats.reset_index()
    merged_df = df.merge(stats, on=['site', 'new_commodity_group'], how='left')

    return merged_df


def export_site_commodity_range_summary(
    df: pd.DataFrame,
    value_columns: list,
    output_path: str,
    output_format: str = "csv"
) -> pd.DataFrame:
    """
    Generates summary stats (min, mean, median, mode, max) for each value column
    grouped by site and new_commodity_group, and saves to file.

    Parameters:
        df (pd.DataFrame): Input data
        value_columns (list): List of columns to compute stats for
        output_path (str): Full path to output file (e.g., 'summary.csv')
        output_format (str): 'csv' or 'excel'

    Returns:
        pd.DataFrame: Summary table as shown in example
    """
    df = df.copy()

    grouped = df.groupby(['site', 'new_commodity_group'])[value_columns]

    # Compute all necessary statistics
    stats_df = grouped.agg(['min', 'mean', 'median', lambda x: x.mode(
    ).iloc[0] if not x.mode().empty else None, 'max'])

    # Flatten MultiIndex columns
    stats_df.columns = [
        f"{stat} of {col}" if stat != "<lambda>" else f"Mode of {col}"
        for col, stat in stats_df.columns
    ]

    stats_df = stats_df.reset_index()

    # Save to file
    if output_format == "csv":
        stats_df.to_csv(output_path, index=False)
    elif output_format == "excel":
        stats_df.to_excel(output_path, index=False)
    else:
        raise ValueError("Invalid output_format. Use 'csv' or 'excel'.")

    return stats_df
