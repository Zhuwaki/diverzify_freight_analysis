
# Vendor Cost Curve Generator
# ---------------------------
# This notebook builds stepwise cost curves for any site + commodity pair
# using a single rates file and plots the result.

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration ---
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

# --- Step 1: Load Rates Table ---
rates_df = pd.read_csv("freight_rates_updated.csv")
rates_df.columns = rates_df.columns.str.strip()

# --- Step 2: Melt Rates Table into Long Format ---
melted_rates = rates_df.melt(
    id_vars=['siteid', 'commodity_group'],
    value_vars=[col for col in rates_df.columns if col not in ['siteid', 'commodity_group']],
    var_name='tier',
    value_name='rate_per_unit'
)

# --- Step 3: Build Tier Mapping ---
tier_rate_map = defaultdict(dict)
for _, row in melted_rates.iterrows():
    key = (row['siteid'], row['commodity_group'])
    tier_rate_map[key][row['tier']] = row['rate_per_unit']

# --- Step 4: Define Rate Lookup Function ---
def get_vendor_rate(siteid, commodity, qty):
    tier = max((t for t, cutoff in TIER_BREAKPOINTS.items() if qty >= cutoff), key=lambda x: TIER_BREAKPOINTS[x])
    return tier_rate_map.get((siteid, commodity), {}).get(tier, 0)

# --- Step 5: Build Cost Curve Function ---
def generate_cost_curve(siteid, commodity, max_qty=40000):
    curve = []
    for q in range(0, max_qty + 1):
        rate = get_vendor_rate(siteid, commodity, q)
        curve.append({
            'siteid': siteid,
            'commodity_group': commodity,
            'quantity_sqyd': q,
            'vendor_rate': rate,
            'vendor_cost': q * rate
        })
    return pd.DataFrame(curve)

# --- Step 6: Example Usage ---
site = "DIT"
commodity = "1CBL"
curve_df = generate_cost_curve(site, commodity)

# --- Step 7: Plot the Stepwise Cost Curve ---
plt.figure(figsize=(10, 6))
plt.step(curve_df['quantity_sqyd'], curve_df['vendor_cost'], where='post', label=f"{site} / {commodity}")
plt.title(f"Vendor Stepwise Cost Curve ({site} / {commodity})")
plt.xlabel("Quantity (SQYD)")
plt.ylabel("Vendor Freight Cost")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
