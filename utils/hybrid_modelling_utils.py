import pandas as pd
import numpy as np
import logging
from utils.model_params_utils import get_freight_class, get_freight_rate

# === Max load capacity constraints ===
MAX_LOAD_LIMITS = {
    '1CBL': 10000,   # SQYD
    '1CPT': 6000,    # SQYD
    '1VNL': 43000    # LBS
}

# === Preferred freight unit per group ===
FREIGHT_UNITS = {
    '1CBL': 'SQYD',
    '1CPT': 'SQYD',
    '1VNL': 'CWT'   # not LBS in rate table
}

logging.basicConfig(level=logging.INFO)


def apply_hybrid_freight_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    hybrid_rows = []
    for _, row in df.iterrows():
        group = row['new_commodity_group']
        site = row['site']
        qty = row['invoice_commodity_quantity']
        original_unit = row['unit']  # This may be incorrect for VNL
        ftl_cost = row['ftl_cost']

        max_qty = MAX_LOAD_LIMITS.get(group)
        freight_unit = FREIGHT_UNITS.get(group, original_unit)

        if pd.isna(max_qty) or pd.isna(qty):
            row['hybrid_overflow_flag'] = np.nan
            row['hybrid_overflow_qty'] = np.nan
            row['hybrid_overflow_cost'] = np.nan
            row['hybrid_ftl_cost_component'] = np.nan
            row['hybrid_cost_total'] = row['xgs_threshold_cost']
            row['hybrid_num_truckloads'] = np.nan
            row['hybrid_overflow_class'] = None
            row['hybrid_overflow_rate'] = np.nan
            hybrid_rows.append(row)
            continue

        # ✅ If qty fits within one full truck, assign only FTL cost
        if qty <= max_qty:
            row['hybrid_overflow_flag'] = False
            row['hybrid_num_truckloads'] = 1
            row['hybrid_overflow_qty'] = 0
            row['hybrid_overflow_cost'] = 0
            row['hybrid_ftl_cost_component'] = ftl_cost
            row['hybrid_cost_total'] = ftl_cost
            row['hybrid_overflow_class'] = None
            row['hybrid_overflow_rate'] = None
            hybrid_rows.append(row)
            continue

        # Number of full truckloads
        full_loads = int(qty // max_qty)
        overflow_qty = qty % max_qty

        total_ftl_cost = full_loads * ftl_cost
        overflow_cost = 0
        overflow_class = None
        overflow_rate = np.nan

        if overflow_qty > 0:
            try:
                freight_class = get_freight_class(round(overflow_qty))
                overflow_class = freight_class
                logging.info(
                    f"📦 Overflow Qty: {overflow_qty} → Freight Class: {freight_class}")

                rate_details, error = get_freight_rate(
                    site, freight_unit, group, freight_class)

                if error or rate_details is None:
                    logging.warning(
                        f"❌ Rate lookup failed for overflow ({site}, {freight_unit}, {group}, {freight_class}): {error}")
                    overflow_cost = np.nan
                    overflow_rate = np.nan
                else:
                    overflow_rate = rate_details['final_rate']
                    min_charge = rate_details.get('minimum_charge', 0)
                    if freight_unit == 'CWT':
                        overflow_rate /= 100
                    logging.info(
                        f"💲 Overflow Rate for {overflow_class} @ {freight_unit} = {overflow_rate}")
                    raw_cost = overflow_qty * overflow_rate
                    overflow_cost = round(max(raw_cost, min_charge), 2)

            except Exception as e:
                logging.error(f"💥 Exception during overflow logic: {e}")
                overflow_cost = np.nan
                overflow_rate = np.nan

        if pd.notna(overflow_cost):
            hybrid_cost = round(total_ftl_cost + overflow_cost, 2)
        else:
            hybrid_cost = np.nan

        row['hybrid_overflow_flag'] = True
        row['hybrid_num_truckloads'] = full_loads
        row['hybrid_overflow_qty'] = overflow_qty
        row['hybrid_overflow_cost'] = overflow_cost
        row['hybrid_ftl_cost_component'] = total_ftl_cost
        row['hybrid_cost_total'] = hybrid_cost
        row['hybrid_overflow_class'] = overflow_class
        row['hybrid_overflow_rate'] = overflow_rate

        hybrid_rows.append(row)

    return pd.DataFrame(hybrid_rows)
