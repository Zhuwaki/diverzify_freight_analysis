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
    df = df.copy()
    df = standardize_input_data(df)

    PRIORITY_COMMODITIES = ['1CBL', '1VNL', '1CPT']

    def classify_standardization_status(group):
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

    def summarize_errors(errors: pd.Series) -> str:
        unique_errors = errors.dropna().unique()
        return '; '.join(unique_errors) if len(unique_errors) > 0 else None

    def summarize_priority_errors(group: pd.DataFrame) -> str:
        priority_errors = group.loc[
            (group['new_commodity_group'].isin(PRIORITY_COMMODITIES)) &
            (~group['standardization_error'].str.contains(
                "successful", case=False, na=False))
        ]['standardization_error']
        unique_priority_errors = priority_errors.dropna().unique()
        return '; '.join(unique_priority_errors) if len(unique_priority_errors) > 0 else "No priority errors found"

    status_df = df.groupby('invoice_id').apply(
        classify_standardization_status).reset_index()
    status_df.columns = ['invoice_id', 'standardization_status']

    invoice_quantity_df = df.groupby(['invoice_id', 'site', 'new_commodity_group'], as_index=False).agg({
        'standard_quantity': 'sum',
        'multiple_commodities': 'first',
        'priority_multiple_commodities': 'first',
        'freight_per_invoice': 'first',
    }).rename(columns={'standard_quantity': 'invoice_commodity_quantity'})

    error_summary_df = df.groupby('invoice_id').agg({
        'standardization_error': summarize_errors
    }).reset_index().rename(columns={'standardization_error': 'error_summary'})

    priority_error_summary_df = df.groupby('invoice_id').apply(
        summarize_priority_errors).reset_index()
    priority_error_summary_df.columns = [
        'invoice_id', 'priority_failure_reasons']

    invoice_df = invoice_quantity_df.merge(
        status_df, on='invoice_id', how='left')
    invoice_df = invoice_df.merge(
        error_summary_df, on='invoice_id', how='left')
    invoice_df = invoice_df.merge(
        priority_error_summary_df, on='invoice_id', how='left')

    def determine_method_unit(group: str) -> tuple:
        group = group.upper()
        if group == '1VNL':
            return 'CWT', 'LBS', 'CWT'
        elif group in ['1CBL', '1CPT']:
            return 'AREA', 'SQYD', 'SQYD'
        else:
            return 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

    invoice_df[['method', 'unit', 'rate_unit']] = invoice_df['new_commodity_group'].apply(
        lambda x: pd.Series(determine_method_unit(x)))

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
                logging.warning(
                    f"⚠️ Skipping invoice {invoice_id}: method={method}, unit={unit}, qty={qty}")
                enriched_rows.append({**row, 'freight_class': None, 'applied_rate': None,
                                     'shipment_type': None, 'invoice_freight_commodity_cost': None})
                continue

            try:
                freight_class = get_freight_class(qty)
            except ValueError as e:
                logging.warning(
                    f"❌ Invalid quantity for freight class on invoice {invoice_id}: {e}")
                enriched_rows.append({**row, 'freight_class': None, 'applied_rate': None,
                                     'shipment_type': None, 'invoice_freight_commodity_cost': None})
                continue

            rate_details, error = get_freight_rate(
                site, rate_unit, group, freight_class)
            if error:
                logging.warning(
                    f"❌ Rate lookup failed for invoice {invoice_id}: {error}")
                enriched_rows.append({**row, 'freight_class': freight_class, 'applied_rate': None, 'shipment_type': None,
                                     'raw_invoice_cost': None, 'invoice_freight_commodity_cost': None, 'minimum_applied': None})
                continue

            base_rate = rate_details['base_rate']
            inflation_rate = rate_details['inflation_rate']
            fsc_rate = rate_details['fsc_rate']
            xgs_rebate = rate_details['xgs_rebate']
            intertim_rate = rate_details['fsc_xgs_rebate']
            star_net_rebate = rate_details['star_net_rebate']
            rate = rate_details['final_rate']

            if method == 'CWT':
                base_rate /= 100
                inflation_rate /= 100
                fsc_rate /= 100
                xgs_rebate /= 100
                intertim_rate /= 100
                star_net_rebate /= 100
                rate /= 100

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
                'base_rate': base_rate,
                'inflation_rate': inflation_rate,
                'fsc_rate': fsc_rate,
                'intertim_rate': intertim_rate,
                'xgs_rebate': xgs_rebate,
                'star_net_rebate': star_net_rebate,
                'applied_rate': rate,
                'shipment_type': shipment_type,
                'raw_invoice_cost': raw_invoice_cost,
                'invoice_freight_commodity_cost': invoice_freight_commodity_cost,
                'minimum_applied': minimum_applied,

            })

        except Exception as e:
            logging.error(f"💥 Unexpected error for invoice {invoice_id}: {e}")
            enriched_rows.append({**row, 'freight_class': None, 'rate': None,
                                 'shipment_type': None, 'invoice_freight_commodity_cost': None})

    enriched_df = pd.DataFrame(enriched_rows)
    return enriched_df


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
