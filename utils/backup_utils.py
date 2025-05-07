# === Global Toggles ===
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from utils.loaders import load_rate_table_from_csv, load_conversion_table

SURCHARGE_DISCOUNT = 0.30


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
    df['est_market_freight_costs'] = df[column] / (1 + SURCHARGE_DISCOUNT)
    return df


def compute_total_freight_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and appends a new column 'total_quantity' at invoice level,
    summing standardized quantity across all rows per invoice.

    Assumes:
    - 'invoice_id' and 'standard_quantity' columns exist.

    Returns:
    - df with new column 'total_quantity'
    """
    if 'invoice_id' not in df.columns or 'est_standard_quantity' not in df.columns:
        raise ValueError(
            "Both 'invoice_id' and 'standard_quantity' must be present in the DataFrame")

    invoice_totals = df.groupby(['invoice_id', 'est_standard_uom'])[
        'est_standard_quantity'].sum().reset_index()
    invoice_totals.rename(
        columns={'est_standard_quantity': 'est_total_quantity'}, inplace=True)  # includes std_uom

    return df.merge(invoice_totals, on=['invoice_id', 'est_standard_uom'], how='left')


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
    import numpy as np

    # Ensure required columns exist
    required_cols = ["invoice_id",
                     "est_market_freight_costs", "est_total_quantity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing values for aggregation
    valid_df = df.dropna(subset=required_cols)

    # Group and calculate invoice-level totals
    invoice_summary = valid_df.groupby("invoice_id").agg(
        est_total_quantity=("est_total_quantity", "first"),
        est_market_freight_costs=("est_market_freight_costs", "first")
    ).reset_index()

    # Avoid divide-by-zero
    invoice_summary["est_market_rate"] = np.where(
        invoice_summary["est_total_quantity"] > 0,
        invoice_summary["est_market_freight_costs"] /
        invoice_summary["est_total_quantity"],
        np.nan
    )

    # Merge back to main dataframe
    df = df.merge(invoice_summary[[
                  "invoice_id", "est_market_rate"]], on="invoice_id", how="left")

    return df


def compute_invoice_freight_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines invoice-level freight class, rate, rate unit, and shipment type (LTL/FTL).

    Assumes:
    - 'invoice_id', 'total_standard_quantity', 'site', 'est_commodity_group', 'est_standard_uom' exist

    Returns:
    - df: merged with invoice-level freight_class, rate, rate_unit, and shipment_type
    """
    results = []

    for _, row in df.iterrows():
        try:
            site = row["site"]
            group = row["est_commodity_group"]
            total_qty = row["est_total_quantity"]
            std_uom = row["est_standard_uom"]

            # Determine method and rate unit
            method = "CWT" if group == "1VNL" else "AREA"
            rate_unit = "CWT" if method == "CWT" else "SQYD"

            # Freight class based on total standardized quantity
            freight_class = get_freight_class(total_qty)

            # Rate lookup
            rate, error = get_freight_rate(
                site, rate_unit, group, freight_class)
            if error:
                raise ValueError(error)

            # Shipment classification
            shipment_type = classify_shipment_by_uom(
                total_qty, "LBS" if method == "CWT" else "SQYD")

        except Exception as e:
            freight_class = None
            rate = None
            rate_unit = None
            shipment_type = None

        results.append({
            "invoice_id": row["invoice_id"],
            "est_freight_class": freight_class,
            "est_xgs_rate": rate/100 if method == "CWT" else rate,
            "est_rate_unit": "$/lbs" if method == "CWT" else "$/SQYD",
            "est_shipment_type": shipment_type
        })

    return df.merge(pd.DataFrame(results), on="invoice_id", how="left")


def compute_xgs_invoice_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes:
    - xgs_total_invoice_raw_cost = invoice_xgs_rate * est_total_quantity
    - xgs_total_invoice_normalised_cost = max(raw_cost, minimum_charge)
    Adds a flag 'xgs_min_applied' if minimum was used.

    Assumes:
    - 'invoice_xgs_rate', 'est_total_quantity', 'site', 'est_commodity_group' exist

    Returns:
    - df with 3 new columns added
    """
    def compute_cost(row):
        try:
            rate = row['est_xgs_rate']
            qty = row['est_total_quantity']
            site = row['site'].upper()
            group = row['est_commodity_group'].upper()

            raw_cost = round(rate * qty, 2)
            min_charge = minimum_charges.get(site, {}).get(group, 0)

            norm_cost = max(
                raw_cost, min_charge) if APPLY_MINIMUM_CHARGES else raw_cost
            min_applied = raw_cost < min_charge if APPLY_MINIMUM_CHARGES else False

            return pd.Series({
                "est_xgs_total_raw_cost": raw_cost,
                "est_xgs_total_normalised_cost": norm_cost,
                "est_xgs_min_applied": min_applied
            })
        except Exception as e:
            return pd.Series({
                "est_xgs_total_raw_cost": None,
                "est_xgs_total_normalised_cost": None,
                "est_xgs_min_applied": False
            })

    df = pd.concat([df, df.apply(compute_cost, axis=1)], axis=1)

    # Calculate beta_xgs_rate: cost per unit
    df["est_normalised_xgs_rate"] = df.apply(
        lambda row: round(
            row["est_xgs_total_normalised_cost"] / row["est_total_quantity"], 4)
        if pd.notnull(row["est_xgs_total_normalised_cost"]) and row["est_total_quantity"] > 0 else None,
        axis=1
    )

    return df


# === Output Filter Function ===
def freight_model_output(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_keep = [
        "site", "site_description", "supplier_no", "supplier_name", "invoice_id",
        "account", "account_description", "ship_to_zip", "po_no", "part_no",
        "part_description", "inv_uom", "invoiced_line_qty", "est_commodity_group",
        "est_method_used", "est_standard_quantity", "est_standard_uom", "est_lbs_per_uom",
        "est_market_freight_costs", "est_total_quantity", "est_market_rate",
        "est_freight_class", "est_xgs_rate", "est_rate_unit", "est_shipment_type",
        "est_xgs_total_raw_cost", "est_xgs_total_normalised_cost",
        "est_normalised_xgs_rate", "est_xgs_min_applied", "market_cost_outlier"
    ]
    return df[[col for col in columns_to_keep if col in df.columns]]


# === Outlier Detection Using IQR ===
def flag_market_cost_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "est_market_freight_costs" not in df.columns:
        raise ValueError(
            "Column 'est_market_freight_costs' is missing from the DataFrame.")

    q1 = df["est_market_freight_costs"].quantile(0.25)
    q3 = df["est_market_freight_costs"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df["market_cost_outlier"] = df["est_market_freight_costs"].apply(
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
        lambda row: row["est_market_freight_costs"] /
        row["est_xgs_total_raw_cost"]
        if pd.notnull(row["est_market_freight_costs"]) and pd.notnull(row["est_xgs_total_raw_cost"]) and row["est_xgs_total_raw_cost"] != 0 else None,
        axis=1
    )

    df["freight_ratio_normal"] = df.apply(
        lambda row: row["est_market_freight_costs"] /
        row["est_xgs_total_normalised_cost"]
        if pd.notnull(row["est_market_freight_costs"]) and pd.notnull(row["est_xgs_total_normalised_cost"]) and row["est_xgs_total_normalised_cost"] != 0 else None,
        axis=1
    )

    df["rate_ratio_raw"] = df.apply(
        lambda row: row["est_market_rate"] / row["est_xgs_rate"]
        if pd.notnull(row["est_market_rate"]) and pd.notnull(row["est_xgs_rate"]) and row["est_xgs_rate"] != 0 else None,
        axis=1
    )

    df["rate_ratio_normal"] = df.apply(
        lambda row: row["est_market_rate"] / row["est_normalised_xgs_rate"]
        if pd.notnull(row["est_market_rate"]) and pd.notnull(row["est_normalised_xgs_rate"]) and row["est_normalised_xgs_rate"] != 0 else None,
        axis=1
    )

    return df
