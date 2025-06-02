# === invoice_freight_utils.py (Patched optimal cost logic for fallback anomalies) ===

import pandas as pd
import numpy as np
from utils.model_params_utils import load_rate_table_from_csv, RATES_CSV_PATH
rate_table = load_rate_table_from_csv(RATES_CSV_PATH)
INFLATION = 1.06  # Inflation factor for cost adjustments
XGS_FTL_REBATE = 0.2  # Rebate factor for FTL costs
STARNET_REBATE = 0.025  # Rebate factor for Starnet costs


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

    def get_ftl_rate(site, unit, commodity_group):
        try:
            return rate_table[site.upper()][unit.upper()][commodity_group.upper()]['__meta__']['ftl_flat_rate']
        except KeyError:
            return np.nan

    required_cols = [
        'invoice_commodity_quantity', 'new_commodity_group', 'applied_rate', 'unit',
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

    df['ftl_cost'] = df.apply(
        lambda row: get_ftl_rate(row['site'], row['unit'], row['new_commodity_group']), axis=1
    )

    df['ftl_cost'] = df['ftl_cost']/INFLATION
    # df['ftl_cost'] = df['ftl_cost']*(1-XGS_FTL_REBATE)
    df['ftl_cost'] = df['ftl_cost']*(1-STARNET_REBATE)

    def fallback_logic(row):
        threshold = vendor_thresholds.get(row['new_commodity_group'], np.inf)
        if row['applied_rate'] == 0:
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
