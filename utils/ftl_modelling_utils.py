# === invoice_freight_utils.py (Patched optimal cost logic for fallback anomalies) ===

import pandas as pd
import numpy as np


def simulate_freight_cost_models_revised(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 3 freight cost simulations:
    - ltl_only_cost: Raw LTL cost if only LTL were used
    - ltl_fallback_cost: Adjusted LTL cost for modeling (accounts for fallback when rate = 0)
    - xgs_threshold_cost: XGS method using vendor threshold rules
    - optimal_cost: Theoretical best (min(FTL, LTL), ignoring thresholds)

    Uses static FTL cost per site based on vendor estimates.
    Now includes logic to catch and correctly handle fallback anomalies.
    """
    df = df.copy()

    vendor_thresholds = {
        '1CBL': 2200, '1CPT': 2200, '1VNL': 20000
    }

    site_ftl_costs = {
        'DIT': 1748.73, 'SPW': 2313.67, 'SPN': 724.83,
        'SPCP': 2160, 'SPT': 2313.67, 'PVF': 3474.81,
        'SPHU': 2160, 'SPTM': 3926.47, 'FSU': 1362.91,
        'CTS': 4405.89, 'SPJ': 2313.67
    }

    required_cols = [
        'invoice_commodity_quantity', 'new_commodity_group', 'rate', 'unit',
        'raw_invoice_cost', 'invoice_freight_commodity_cost', 'minimum_applied', 'site'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df['ltl_only_cost'] = np.where(
        df['minimum_applied'],
        df['invoice_freight_commodity_cost'],
        df['raw_invoice_cost']
    )

    df['ftl_cost'] = df['site'].map(site_ftl_costs)

    def fallback_logic(row):
        threshold = vendor_thresholds.get(row['new_commodity_group'], np.inf)
        if row['rate'] == 0:
            if row['invoice_commodity_quantity'] >= threshold:
                return row['ftl_cost']
            else:
                return row['ltl_only_cost']
        else:
            return row['ltl_only_cost']

    df['ltl_fallback_cost'] = df.apply(fallback_logic, axis=1)

    def xgs_cost(row):
        threshold = vendor_thresholds.get(row['new_commodity_group'], np.inf)
        return min(row['ftl_cost'], row['ltl_fallback_cost']) if row['invoice_commodity_quantity'] >= threshold else row['ltl_fallback_cost']

    def xgs_method(row):
        threshold = vendor_thresholds.get(row['new_commodity_group'], np.inf)
        if row['invoice_commodity_quantity'] >= threshold:
            return 'FTL' if row['ftl_cost'] <= row['ltl_fallback_cost'] else 'LTL'
        else:
            return 'LTL'

    df['xgs_threshold_cost'] = df.apply(xgs_cost, axis=1)
    df['xgs_method'] = df.apply(xgs_method, axis=1)

    def robust_optimal(row):
        ftl = row['ftl_cost']
        ltl = row['ltl_fallback_cost']
        if pd.notnull(ftl) and pd.notnull(ltl):
            return min(ftl, ltl)
        elif pd.notnull(ltl):
            return ltl
        elif pd.notnull(ftl):
            return ftl
        else:
            return np.nan

    def robust_opt_method(row):
        ftl = row['ftl_cost']
        ltl = row['ltl_fallback_cost']
        if pd.notnull(ftl) and pd.notnull(ltl):
            return 'FTL' if ftl <= ltl else 'LTL'
        elif pd.notnull(ftl):
            return 'FTL'
        elif pd.notnull(ltl):
            return 'LTL'
        else:
            return 'N/A'

    df['optimal_cost'] = df.apply(robust_optimal, axis=1)
    df['optimal_method'] = df.apply(robust_opt_method, axis=1)

    df['ftl_blocked'] = (df['xgs_method'] == 'LTL') & (
        df['optimal_method'] == 'FTL')
    df['penalty_vs_optimal'] = df['xgs_threshold_cost'] - df['optimal_cost']

    return df
