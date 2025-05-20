import pandas as pd
import numpy as np

# Vendor FTL thresholds (billing-based)
VENDOR_THRESHOLDS = {
    '1CBL': 2200,
    '1CPT': 2200,
    '1VNL': 20000  # in LBS
}

# Actual max truck capacities (physical limits)
MAX_CAPACITIES = {
    '1CBL': 10000,   # SQYD
    '1CPT': 6000,    # SQYD
    '1VNL': 43000    # LBS
}


def evaluate_vendor_vs_physical_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def evaluate_row(row):
        group = row['new_commodity_group']
        qty = row['invoice_commodity_quantity']
        ftl_cost = row.get('ftl_cost', np.nan)

        vendor_thresh = VENDOR_THRESHOLDS.get(group, np.nan)
        max_capacity = MAX_CAPACITIES.get(group, np.nan)

        if pd.isna(qty) or pd.isna(ftl_cost) or pd.isna(vendor_thresh) or pd.isna(max_capacity):
            return pd.Series({
                'ftl_trigger_threshold': vendor_thresh,
                'max_physical_capacity': max_capacity,
                'ftl_utilization_pct': np.nan,
                'unused_capacity_qty': np.nan,
                'unused_capacity_pct': np.nan,
                'cost_per_unit_shipped': np.nan,
                'ideal_cost_per_unit': np.nan,
                'cost_penalty_per_unit': np.nan,
                'consolidation_opportunity': np.nan,
                'consolidation_guidance': 'Insufficient data'
            })

        if qty >= vendor_thresh:
            utilization_pct = qty / max_capacity
            unused_qty = max_capacity - qty
            unused_pct = unused_qty / max_capacity
            cost_per_unit = ftl_cost / qty
            ideal_cost = ftl_cost / max_capacity
            penalty = cost_per_unit - ideal_cost

            consolidate = qty < max_capacity
            guidance = "Hold for consolidation" if consolidate else "Proceed to ship (full or LTL)"
        else:
            utilization_pct = 0
            unused_qty = max_capacity
            unused_pct = 1.0
            cost_per_unit = np.nan
            ideal_cost = ftl_cost / max_capacity
            penalty = np.nan

            consolidate = False
            guidance = "Proceed to ship (full or LTL)"

        return pd.Series({
            'ftl_trigger_threshold': vendor_thresh,
            'max_physical_capacity': max_capacity,
            'ftl_utilization_pct': utilization_pct,
            'unused_capacity_qty': unused_qty,
            'unused_capacity_pct': unused_pct,
            'cost_per_unit_shipped': cost_per_unit,
            'ideal_cost_per_unit': ideal_cost,
            'cost_penalty_per_unit': penalty,
            'consolidation_opportunity': consolidate,
            'consolidation_guidance': guidance
        })

    metrics_df = df.apply(evaluate_row, axis=1)
    return pd.concat([df, metrics_df], axis=1)
