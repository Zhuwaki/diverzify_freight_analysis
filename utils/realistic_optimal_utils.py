import pandas as pd
import numpy as np

# === Max load capacity constraints ===
MAX_LOAD_LIMITS = {
    '1CBL': 10000,   # SQYD
    '1CPT': 6000,    # SQYD
    '1VNL': 43000    # LBS
}


def apply_realistic_optimal_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def compute_optimal(row):
        group = row['new_commodity_group']
        qty = row['invoice_commodity_quantity']
        ftl = row.get('ftl_cost', np.nan)
        ltl = row.get('ltl_fallback_cost', np.nan)
        hybrid = row.get('hybrid_cost_total', np.nan)

        max_qty = MAX_LOAD_LIMITS.get(group)

        if pd.isna(qty) or pd.isna(max_qty):
            return pd.Series({'realistic_optimal_cost': np.nan, 'realistic_optimal_method': 'N/A'})

        if qty <= max_qty:
            if pd.notna(ftl) and pd.notna(ltl):
                return pd.Series({
                    'realistic_optimal_cost': min(ftl, ltl),
                    'realistic_optimal_method': 'FTL' if ftl <= ltl else 'LTL'
                })
            elif pd.notna(ftl):
                return pd.Series({'realistic_optimal_cost': ftl, 'realistic_optimal_method': 'FTL'})
            elif pd.notna(ltl):
                return pd.Series({'realistic_optimal_cost': ltl, 'realistic_optimal_method': 'LTL'})
            else:
                return pd.Series({'realistic_optimal_cost': np.nan, 'realistic_optimal_method': 'N/A'})
        else:
            if pd.notna(hybrid) and pd.notna(ltl):
                return pd.Series({
                    'realistic_optimal_cost': min(hybrid, ltl),
                    'realistic_optimal_method': 'HYBRID' if hybrid <= ltl else 'LTL'
                })
            elif pd.notna(hybrid):
                return pd.Series({'realistic_optimal_cost': hybrid, 'realistic_optimal_method': 'HYBRID'})
            elif pd.notna(ltl):
                return pd.Series({'realistic_optimal_cost': ltl, 'realistic_optimal_method': 'LTL'})
            else:
                return pd.Series({'realistic_optimal_cost': np.nan, 'realistic_optimal_method': 'N/A'})

    optimal_df = df.apply(compute_optimal, axis=1)
    return pd.concat([df, optimal_df], axis=1)
