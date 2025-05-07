
# Freight Cost Integration Notebook
# ---------------------------------
# This notebook processes invoice data and merges it with a vendor freight rate curve,
# preserving all rows and logging key transformation steps.

import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
VENDOR_TIERS = {
    'L5C': 0.4386,
    '5C': 0.4308,
    '1M': 0.423,
    '2M': 0.4153,
    '3M': 0.4075,
    '5M': 0.3966,
    '10M': 0.3862
}
TIER_BREAKPOINTS = {
    'L5C': 0,
    '5C': 500,
    '1M': 1000,
    '2M': 2000,
    '3M': 3000,
    '5M': 5000,
    '10M': 10000,
    '20M': 20000
}

# --- Functions ---

def load_invoice_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['invoice_row_id'] = df.index + 1
    df['invoice_commodity_quantity'] = pd.to_numeric(df['invoice_commodity_quantity'], errors='coerce')
    df['market_freight_costs'] = pd.to_numeric(df['market_freight_costs'].str.replace(",", ""), errors='coerce')
    df['ltl_only_cost'] = pd.to_numeric(df['ltl_only_cost'].str.replace(",", ""), errors='coerce')
    df['optimal_cost'] = pd.to_numeric(df['optimal_cost'].str.replace(",", ""), errors='coerce')
    logging.info(f"Loaded invoice data: {df.shape[0]} rows.")
    return df

def get_vendor_rate(qty):
    applicable_tier = max((tier for tier, limit in TIER_BREAKPOINTS.items() if qty >= limit), key=lambda x: TIER_BREAKPOINTS[x])
    return VENDOR_TIERS.get(applicable_tier, 0)

def process_invoice_data(df):
    df['vendor_rate'] = df['invoice_commodity_quantity'].apply(get_vendor_rate)
    df['vendor_estimated_cost'] = df['invoice_commodity_quantity'] * df['vendor_rate']
    df['quantity_sqyd'] = df['invoice_commodity_quantity']
    df['source'] = 'actual'
    logging.info("Processed invoice data with vendor rates.")
    return df

def build_vendor_cost_curve(max_qty=40000):
    rows = []
    for qty in range(0, max_qty + 1):
        rate = get_vendor_rate(qty)
        rows.append({
            'invoice_row_id': 0,
            'quantity_sqyd': qty,
            'vendor_rate': rate,
            'vendor_estimated_cost': qty * rate,
            'actual_freight_cost': None,
            'ltl_only_cost': None,
            'optimal_cost': None,
            'source': 'vendor'
        })
    df = pd.DataFrame(rows)
    logging.info("Built vendor cost curve.")
    return df

def merge_and_export(vendor_df, invoice_df, output_path):
    invoice_export = invoice_df[[
        'invoice_row_id', 'quantity_sqyd', 'vendor_rate', 'vendor_estimated_cost',
        'market_freight_costs', 'ltl_only_cost', 'optimal_cost', 'source'
    ]].rename(columns={'market_freight_costs': 'actual_freight_cost'})

    final_df = pd.concat([vendor_df, invoice_export], ignore_index=True)
    final_df.to_excel(output_path, index=False)
    logging.info(f"Exported combined file to: {output_path}")
    return final_df

# --- Execution Example ---
# file_path = '1cbl_dit_model_output.csv'
# output_path = 'final_combined_output.xlsx'
# df_invoice = load_invoice_data(file_path)
# df_invoice_processed = process_invoice_data(df_invoice)
# df_vendor = build_vendor_cost_curve()
# final_combined = merge_and_export(df_vendor, df_invoice_processed, output_path)
