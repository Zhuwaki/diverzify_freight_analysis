
# Unified Freight Cost Curve + Invoice Integration Notebook
# ---------------------------------------------------------
# This notebook generates vendor cost curves by site + commodity
# and merges them with invoice data for multi-site freight analysis.

import pandas as pd
import matplotlib.pyplot as plt
import logging
from collections import defaultdict

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Config ---
TIER_BREAKPOINTS = {
    'L5C': 0,
    '5C': 500,
    '1M': 1000,
    '2M': 2000,
    '3M': 3000,
    '5M': 5000,
    '10M': 10000,
    '20M': 20000,
    '30M': 30000,
    '40M': 40000
}

# --- Load and Structure Vendor Rates ---
def load_and_prepare_rates(rates_file):
    rates_df = pd.read_csv(rates_file)
    rates_df.columns = rates_df.columns.str.strip()
    melted = rates_df.melt(
        id_vars=['siteid', 'commodity_group'],
        value_vars=[col for col in rates_df.columns if col not in ['siteid', 'commodity_group']],
        var_name='tier',
        value_name='rate_per_unit'
    )
    rate_map = defaultdict(dict)
    for _, row in melted.iterrows():
        key = (row['siteid'], row['commodity_group'])
        rate_map[key][row['tier']] = row['rate_per_unit']
    logging.info("Vendor rate table loaded and transformed.")
    return rate_map

# --- Rate Lookup ---
def get_vendor_rate(rate_map, siteid, commodity, qty):
    tier = max((t for t, cutoff in TIER_BREAKPOINTS.items() if qty >= cutoff), key=lambda x: TIER_BREAKPOINTS[x])
    return rate_map.get((siteid, commodity), {}).get(tier, 0)

# --- Build Vendor Cost Curve ---
def generate_cost_curve(rate_map, siteid, commodity, max_qty=40000):
    curve = []
    for q in range(0, max_qty + 1):
        rate = get_vendor_rate(rate_map, siteid, commodity, q)
        curve.append({
            'siteid': siteid,
            'commodity_group': commodity,
            'quantity_sqyd': q,
            'vendor_rate': rate,
            'vendor_estimated_cost': q * rate,
            'source': 'vendor'
        })
    return pd.DataFrame(curve)

# --- Load and Process Invoice Data ---
def load_invoice_data(invoice_file):
    df = pd.read_csv(invoice_file)
    df.columns = df.columns.str.strip()
    df['invoice_row_id'] = df.index + 1
    df['invoice_commodity_quantity'] = pd.to_numeric(df['invoice_commodity_quantity'], errors='coerce')
    df['market_freight_costs'] = pd.to_numeric(df['market_freight_costs'].str.replace(",", ""), errors='coerce')
    df['ltl_only_cost'] = pd.to_numeric(df['ltl_only_cost'].str.replace(",", ""), errors='coerce')
    df['optimal_cost'] = pd.to_numeric(df['optimal_cost'].str.replace(",", ""), errors='coerce')
    df['quantity_sqyd'] = df['invoice_commodity_quantity']
    df['source'] = 'actual'
    logging.info(f"Invoice file loaded with {df.shape[0]} rows.")
    return df

# --- Merge and Export Output ---
def merge_and_export(vendor_curve_df, invoice_df, siteid, commodity, output_dir):
    filtered_invoice = invoice_df[
        (invoice_df['siteid'] == siteid) & 
        (invoice_df['commodity_group'] == commodity)
    ].copy()

    if filtered_invoice.empty:
        logging.warning(f"No matching invoice data for {siteid} / {commodity}")
        return

    invoice_export = filtered_invoice[[
        'invoice_row_id', 'quantity_sqyd', 'market_freight_costs',
        'ltl_only_cost', 'optimal_cost'
    ]].copy()
    invoice_export['vendor_rate'] = None
    invoice_export['vendor_estimated_cost'] = None
    invoice_export['source'] = 'actual'

    invoice_export = invoice_export.rename(columns={'market_freight_costs': 'actual_freight_cost'})

    combined_df = pd.concat([vendor_curve_df, invoice_export], ignore_index=True)
    filename = f"{output_dir}/{siteid}_{commodity}_freight_costs.xlsx"
    combined_df.to_excel(filename, index=False)
    logging.info(f"Exported: {filename}")

# --- Optional Plot ---
def plot_cost_curve(df, siteid, commodity):
    vendor_df = df[df['source'] == 'vendor']
    plt.figure(figsize=(10, 6))
    plt.step(vendor_df['quantity_sqyd'], vendor_df['vendor_estimated_cost'], where='post', label=f"{siteid} / {commodity}")
    plt.title(f"Vendor Stepwise Cost Curve ({siteid} / {commodity})")
    plt.xlabel("Quantity (SQYD)")
    plt.ylabel("Vendor Freight Cost")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Example Driver ---
# rates_file = 'freight_rates_updated.csv'
# invoice_file = 'invoice_data.csv'
# output_dir = 'output'

# rate_map = load_and_prepare_rates(rates_file)
# invoice_df = load_invoice_data(invoice_file)

# for site in invoice_df['siteid'].unique():
#     for commodity in invoice_df[invoice_df['siteid'] == site]['commodity_group'].unique():
#         curve = generate_cost_curve(rate_map, site, commodity)
#         merge_and_export(curve, invoice_df, site, commodity, output_dir)
