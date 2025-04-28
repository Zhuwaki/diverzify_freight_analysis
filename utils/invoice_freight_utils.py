# === invoice_freight_utils.py ===

import pandas as pd
from utils.freight_model_utils import (
    standardize_commodity,
    get_freight_class,
    get_freight_rate,
    classify_shipment_by_uom
)


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
        try:
            result = standardize_commodity(
                quantity=row['invoiced_line_qty'],
                inv_uom=row['inv_uom'],
                commodity_group=row['commodity_group'],
                conversion_code=row['conversion_code'],
                site=row['site']
            )

            if "error" in result:
                error_message = result['error']
                standard_quantity = None
                standard_uom = None
            else:
                error_message = None
                standard_quantity = result['standard_quantity']
                standard_uom = result['standard_uom']
                lbs_per_uom = result['lbs_per_uom']

            standardized_rows.append({
                **row,
                'standard_quantity': standard_quantity,
                'standard_uom': standard_uom,
                'lbs_per_uom': lbs_per_uom,
                'standardization_error': error_message
            })

        except Exception as e:
            standardized_rows.append({
                **row,
                'standard_quantity': None,
                'standard_uom': None,
                'standardization_error': f"Unexpected error: {e}"
            })

    standardized_df = pd.DataFrame(standardized_rows)
    standardized_df.to_csv(
        'data/downloads/standardized_input_data.csv', index=False)
    return standardized_df


def prepare_invoice_freight_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares invoice-level freight data by grouping standardized quantities
    and carrying forward any standardization errors.
    """
    df = df.copy()

    # === Step 1: Standardize input quantities and UOM ===
    df = standardize_input_data(df)

    def classify_standardization_status(group):
        total_rows = len(group)
        failed_rows = group['standardization_error'].notna().sum()

        if failed_rows == total_rows:
            return "ALL_FAILED"
        elif failed_rows > 0:
            return "PARTIALLY_FAILED"
        else:
            return "SUCCESS"

    # Standardize
    df = standardize_input_data(df)

    # Add standardization status BEFORE aggregation
    status_df = df.groupby('invoice_id').apply(
        classify_standardization_status).reset_index()
    status_df.columns = ['invoice_id', 'standardization_status']

    # === Step 2: Aggregate invoice quantities and errors ===
    invoice_df = df.groupby(['invoice_id', 'site', 'commodity_group'], as_index=False).agg({
        'standard_quantity': 'sum',
        'standardization_error': lambda x: '; '.join(x.dropna().unique()) if x.notna().any() else None,
        'multiple_commodities': 'first'
    })
    invoice_df.rename(
        columns={'standard_quantity': 'invoice_commodity_quantity'}, inplace=True)

    invoice_df = invoice_df.merge(status_df, on='invoice_id', how='left')

    # === Step 3: Determine method and unit ===
    def determine_method_unit(group: str) -> tuple:
        group = group.upper()
        if group == '1VNL':
            return 'CWT', 'LBS', 'CWT'  # Method=CWT, Unit=LBS, Rate Unit=CWT
        elif group in ['1CBL', '1CPT']:
            return 'AREA', 'SQYD', 'SQYD'  # Method=AREA, Unit=SQYD, Rate Unit=SQYD
        else:
            return 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'

    invoice_df[['method', 'unit', 'rate_unit']] = invoice_df['commodity_group'].apply(
        lambda x: pd.Series(determine_method_unit(x))
    )

    # === Step 4: Enrich with freight class, rate, shipment type ===
    enriched_rows = []

    for _, row in invoice_df.iterrows():
        try:
            invoice_id = row['invoice_id']
            site = row['site'].upper()
            group = row['commodity_group'].upper()
            qty = row['invoice_commodity_quantity']
            method = row['method']
            unit = row['unit']          # For shipment classification
            rate_unit = row['rate_unit']  # For fetching rates
            error_flag = row.get('standardization_error', None)

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
                                      'invoice_freight_commodity_cost': None})
                continue

            if method == 'CWT':
                rate = rate / 100  # CWT rates are per 100lbs

            shipment_type = classify_shipment_by_uom(qty, unit)
            invoice_freight_commodity_cost = round(rate * qty, 2)

            enriched_rows.append({
                **row,
                'freight_class': freight_class,
                'rate': rate,
                'shipment_type': shipment_type,
                'invoice_freight_commodity_cost': invoice_freight_commodity_cost
            })

        except Exception as e:
            enriched_rows.append({**row,
                                  'freight_class': None,
                                  'rate': None,
                                  'shipment_type': None,
                                  'invoice_freight_commodity_cost': None})

    enriched_df = pd.DataFrame(enriched_rows)

    return enriched_df

# Example usage (in your main model file):
# from utils.invoice_freight_utils import prepare_invoice_freight_summary
# result_df = prepare_invoice_freight_summary(input_df)
