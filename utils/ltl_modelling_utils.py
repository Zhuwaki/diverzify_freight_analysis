# === invoice_freight_utils.py ===

import logging
import pandas as pd
import numpy as np
from utils.model_params_utils import (
    standardize_commodity,
    get_freight_class,
    get_freight_rate,
    classify_shipment_by_uom,
    minimum_charges,
)


# === Adjustable Discount Rates ===
SURCHARGE_DISCOUNT = 0


def standardize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes invoiced quantity and unit of measure using the standardize_commodity function.
    Adds:
    - 'standard_quantity'
    - 'standard_uom'
    - 'standardization_error' (new: logs any errors at row-level)
    """
    standardized_rows = []
    for idx, row in df.iterrows():
        result = standardize_commodity(
            quantity=row['invoiced_line_qty'],
            inv_uom=row['inv_uom'],
            commodity_group=row['new_commodity_group'],
            conversion_code=row['conversion_code'],
            site=row['site']
        )

        standardized_rows.append({
            **row,
            'standard_quantity': result['standard_quantity'],
            'standard_uom': result['standard_uom'],
            'lbs_per_uom': result['lbs_per_uom'],
            'standardization_error': result['standardization_error']
        })

    standardized_df = pd.DataFrame(standardized_rows)
    standardized_df.to_csv(
        'data/downloads/standardized_input_data.csv', index=False)
    return standardized_df


def estimate_invoice_freight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares invoice-level freight data by grouping standardized quantities,
    carrying forward all standardization errors, and separately tracking
    priority commodity standardization failures.
    """
    df = df.copy()

    # === Step 1: Standardize input quantities and UOM ===
    df = standardize_input_data(df)

    # Define priority commodities
    PRIORITY_COMMODITIES = ['1CBL', '1VNL', '1CPT']

    # === Step 2: Classify standardization status ===
    def classify_standardization_status(group):
        """
        Classifies invoice based only on priority commodity standardization status.
        """
        priority_group = group[group['new_commodity_group'].isin(
            PRIORITY_COMMODITIES)]

        if priority_group.empty:
            return "NO_PRIORITY"

        successful_rows = priority_group['standardization_error'].str.contains(
            "successful", case=False, na=False).sum()

        total_priority_rows = len(priority_group)

        if successful_rows == 0:
            return "ALL_FAILED"
        elif successful_rows < total_priority_rows:
            return "PARTIALLY_FAILED"
        else:
            return "SUCCESS"

    # === Step 3: Summarize all errors ===
    def summarize_errors(errors: pd.Series) -> str:
        """
        Joins unique error messages into a single comma-separated string.
        """
        unique_errors = errors.dropna().unique()
        return '; '.join(unique_errors) if len(unique_errors) > 0 else None

    # === Step 4: Summarize priority commodity errors only ===
    def summarize_priority_errors(group: pd.DataFrame) -> str:

        priority_errors = group.loc[
            (group['new_commodity_group'].isin(PRIORITY_COMMODITIES)) &
            (~group['standardization_error'].str.contains(
                "successful", case=False, na=False))
        ]['standardization_error']

        unique_priority_errors = priority_errors.dropna().unique()

        if len(unique_priority_errors) > 0:
            return '; '.join(unique_priority_errors)
        else:
            return "No priority errors found"

    # === Step 5: Standardization status by invoice ===
    status_df = df.groupby('invoice_id').apply(
        classify_standardization_status).reset_index()
    status_df.columns = ['invoice_id', 'standardization_status']

    # === Step 6: Aggregate invoice quantities and all error summaries ===
    invoice_quantity_df = df.groupby(['invoice_id', 'site', 'new_commodity_group'], as_index=False).agg({
        'standard_quantity': 'sum',
        'multiple_commodities': 'first',
        'priority_multiple_commodities': 'first',
        'freight_per_invoice': 'first',

    })
    invoice_quantity_df.rename(
        columns={'standard_quantity': 'invoice_commodity_quantity'},
        inplace=True
    )

    # === Step 7: Summarize all error messages per invoice ===
    error_summary_df = df.groupby('invoice_id').agg({
        'standardization_error': summarize_errors
    }).reset_index()
    error_summary_df.rename(
        columns={'standardization_error': 'error_summary'},
        inplace=True
    )

    # === Step 8: Summarize priority commodity failure reasons per invoice ===
    priority_error_summary_df = df.groupby('invoice_id').apply(
        summarize_priority_errors).reset_index()
    priority_error_summary_df.columns = [
        'invoice_id', 'priority_failure_reasons']

    # === Step 9: Merge all summaries together ===
    invoice_df = invoice_quantity_df.merge(
        status_df, on='invoice_id', how='left')
    invoice_df = invoice_df.merge(
        error_summary_df, on='invoice_id', how='left')
    invoice_df = invoice_df.merge(
        priority_error_summary_df, on='invoice_id', how='left')

    # === Step 10: Determine method and unit ===
    def determine_method_unit(group: str) -> tuple:
        group = group.upper()
        if group == '1VNL':
            return 'CWT', 'LBS', 'CWT'
        elif group in ['1CBL', '1CPT']:
            return 'AREA', 'SQYD', 'SQYD'
        else:
            return 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

    invoice_df[['method', 'unit', 'rate_unit']] = invoice_df['new_commodity_group'].apply(
        lambda x: pd.Series(determine_method_unit(x))
    )

    # === Step 11: Enrich with freight class, rate, shipment type ===
    enriched_rows = []

    for _, row in invoice_df.iterrows():
        try:
            invoice_id = row['invoice_id']
            site = row['site'].upper()
            group = row['new_commodity_group'].upper()
            qty = row['invoice_commodity_quantity']
            method = row['method']
            unit = row['unit']
            rate_unit = row['rate_unit']

            if method == 'UNKNOWN' or unit == 'UNKNOWN' or qty is None or qty == 0:
                enriched_rows.append({**row,
                                      'freight_class': None,
                                      'rate': None,
                                      'shipment_type': None,
                                      'invoice_freight_commodity_cost': None})
                continue

            freight_class = get_freight_class(qty)
            rate, error = get_freight_rate(
                site, rate_unit, group, freight_class)

            if error:
                enriched_rows.append({**row,
                                      'freight_class': None,
                                      'rate': None,
                                      'shipment_type': None,
                                      'raw_invoice_cost': None,
                                      'invoice_freight_commodity_cost': None,
                                      'minimum_applied': None})
                continue

            if method == 'CWT':
                rate = rate / 100  # Adjust for per 100lbs

            shipment_type = classify_shipment_by_uom(qty, unit)

            raw_invoice_cost = round(rate * qty, 2)
            min_charge = minimum_charges.get(site, {}).get(group, 0)
            if raw_invoice_cost < min_charge:
                invoice_freight_commodity_cost = min_charge
                minimum_applied = True
            else:
                invoice_freight_commodity_cost = raw_invoice_cost
                minimum_applied = False

            enriched_rows.append({
                **row,
                'freight_class': freight_class,
                'rate': rate,
                'shipment_type': shipment_type,
                'raw_invoice_cost': raw_invoice_cost,
                'invoice_freight_commodity_cost': invoice_freight_commodity_cost,
                'minimum_applied': minimum_applied
            })

        except Exception:
            enriched_rows.append({**row,
                                  'freight_class': None,
                                  'rate': None,
                                  'shipment_type': None,
                                  'invoice_freight_commodity_cost': None})

    enriched_df = pd.DataFrame(enriched_rows)

    return enriched_df


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


def filter_valid_priority_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame for valid priority lines based on:
    - priority_multiple_commodities == False
    - standardization_status == 'SUCCESS'
    - method != 'UNKNOWN'
    """
    filtered_df = df[
        (df['priority_multiple_commodities'] == False) &
        (df['standardization_status'] == "SUCCESS") &
        (df['method'] != "UNKNOWN")
    ].copy()

    return filtered_df
