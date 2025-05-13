import pandas as pd
import numpy as np
import logging

SURCHARGE_DISCOUNT = 0


def calibrate_surcharge(df: pd.DataFrame, column="freight_per_invoice") -> pd.DataFrame:
    print(f"🧪 Adjusting for surcharge")
    """
    Adds a new column with the adjusted market freight rate.

    Parameters:
    - df: input DataFrame
    - column: name of column containing original freight values

    Returns:
    - DataFrame with new column 'adjusted_freight_price'
    """
    logging.info(
        f"✅ Applying market freight discount of {SURCHARGE_DISCOUNT*100:.0f}% to '{column}'...")
    df['historical_market_freight_costs'] = df[column] / \
        (1 + SURCHARGE_DISCOUNT)
    return df


def compute_market_rates(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🧪 Estimating market rates")
    """
    Calculates invoice-level total standard quantity and estimated market rate.
    Adds a 'market_estimated_rate' column based on:
        freight_per_invoice / total_standard_quantity

    Assumes:
    - 'invoice_id' exists
    - 'freight_per_invoice' is populated
    - 'est_standard_quantity' is numeric

    Returns:
    - df: enriched with 'market_estimated_rate'
    """

    # Avoid divide-by-zero
    df["market_rate"] = np.where(
        df["invoice_commodity_quantity"] > 0,
        df["historical_market_freight_costs"] /
        df["invoice_commodity_quantity"],
        np.nan
    )

    df["xgs_applied_rate"] = np.where(
        df["invoice_commodity_quantity"] > 0,
        df["invoice_freight_commodity_cost"] /
        df["invoice_commodity_quantity"],
        np.nan
    )
    df["xgs_raw_rate"] = np.where(
        df["invoice_commodity_quantity"] > 0,
        df["raw_invoice_cost"] /
        df["invoice_commodity_quantity"],
        np.nan
    )

    # Merge back to main dataframe

    return df


def flag_market_cost_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "historical_market_freight_costs" not in df.columns:
        raise ValueError(
            "Column 'est_market_freight_costs' is missing from the DataFrame.")

    q1 = df["historical_market_freight_costs"].quantile(0.25)
    q3 = df["historical_market_freight_costs"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df["market_cost_outlier"] = df["historical_market_freight_costs"].apply(
        lambda x: "LOW" if x < lower_bound else (
            "HIGH" if x > upper_bound else "NORMAL")
    )

    q1 = df["freight_ratio_raw"].quantile(0.25)
    q3 = df["freight_ratio_raw"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df["freight_ratio_raw_outlier"] = df["freight_ratio_raw"].apply(
        lambda x: "LOW" if x < lower_bound else (
            "HIGH" if x > upper_bound else "NORMAL")
    )

    q1 = df["freight_ratio_normal"].quantile(0.25)
    q3 = df["freight_ratio_normal"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df["freight_ratio_normal_outlier"] = df["freight_ratio_normal"].apply(
        lambda x: "LOW" if x < lower_bound else (
            "HIGH" if x > upper_bound else "NORMAL")
    )
    return df


def compute_freight_and_rate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds freight and rate ratio calculations to the DataFrame.
    - freight_ratio_raw = est_market_freight_costs / est_xgs_total_raw_cost
    - freight_ratio_normal = est_market_freight_costs / est_xgs_total_normalised_cost
    - rate_ratio_raw = est_market_rate / est_xgs_rate
    - rate_ratio_normal = est_market_rate / est_normalised_xgs_rate
    """
    df["freight_ratio_raw"] = df.apply(
        lambda row: row["historical_market_freight_costs"] /
        row["raw_invoice_cost"]
        if pd.notnull(row["historical_market_freight_costs"]) and pd.notnull(row["raw_invoice_cost"]) and row["raw_invoice_cost"] != 0 else None,
        axis=1
    )

    df["freight_ratio_normal"] = df.apply(
        lambda row: row["historical_market_freight_costs"] /
        row["invoice_freight_commodity_cost"]
        if pd.notnull(row["historical_market_freight_costs"]) and pd.notnull(row["invoice_freight_commodity_cost"]) and row["invoice_freight_commodity_cost"] != 0 else None,
        axis=1
    )

    df["rate_ratio_raw"] = df.apply(
        lambda row: row["market_rate"] / row["rate"]
        if pd.notnull(row["market_rate"]) and pd.notnull(row["rate"]) and row["rate"] != 0 else None,
        axis=1
    )

    df["rate_ratio_normal"] = df.apply(
        lambda row: row["market_rate"] / row["xgs_applied_rate"]
        if pd.notnull(row["market_rate"]) and pd.notnull(row["xgs_applied_rate"]) and row["xgs_applied_rate"] != 0 else None,
        axis=1
    )

    return df
