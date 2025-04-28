# === freight_estimation_module.py ===

import pandas as pd
import logging
from utils.freight_model_utils import (
    standardize_commodity,
    get_freight_class,
    get_freight_rate,
    minimum_charges,
    APPLY_MINIMUM_CHARGES
)

# Setup logger
logging.basicConfig(level=logging.INFO)


# === STEP 1: Line-Level Freight Estimation with Linked Helpers ===


def estimate_line_invoice_freight_cost(row: pd.Series) -> pd.Series:
    """
    Estimate freight cost for a single invoice line using standardized model logic.
    Returns:
    - xgs_line_raw_cost
    - xgs_line_normalised_cost
    """
    try:
        result = standardize_commodity(
            quantity=row['invoiced_line_qty'],
            inv_uom=row['inv_uom'],
            commodity_group=row['new_commodity_group'],
            conversion_code=row['conversion_code'],
            site=row['site']
        )

        if "error" in result:
            logging.warning(
                f"⚠️ Standardization error on invoice {row.get('invoice_id', 'Unknown')}: {result['error']}")
            return pd.Series({
                "xgs_line_raw_cost": 0.0,
                "xgs_line_normalised_cost": 0.0
            })

        standard_quantity = result['standard_quantity']
        method = result['method_used']
        group = result['commodity_group']
        site = row['site'].upper()

        freight_class = get_freight_class(standard_quantity)

        rate_unit = "CWT" if method == "CWT" else "SQYD"

        rate, error = get_freight_rate(site, rate_unit, group, freight_class)
        if error:
            logging.warning(
                f"⚠️ Rate lookup error on invoice {row.get('invoice_id', 'Unknown')}: {error}")
            return pd.Series({
                "xgs_line_raw_cost": 0.0,
                "xgs_line_normalised_cost": 0.0
            })

        if method == "CWT":
            rate = rate / 100  # CWT adjustment: $/100 lbs

        raw_cost = round(standard_quantity * rate, 2)

        # Apply minimum charge if enabled
        min_charge = minimum_charges.get(site, {}).get(group, 0)
        normalised_cost = max(
            raw_cost, min_charge) if APPLY_MINIMUM_CHARGES else raw_cost

        return pd.Series({
            "xgs_line_raw_cost": raw_cost,
            "xgs_line_normalised_cost": normalised_cost,
            "freight_class": freight_class,
            "method_used": method,
            "rate_unit": rate_unit,
            "standard_quantity": standard_quantity,
            "commodity_group": group,
            "rate": rate,

        })

    except Exception as e:
        logging.error(
            f"❌ Unexpected error estimating freight for invoice {row.get('invoice_id', 'Unknown')}: {e}")
        return pd.Series({
            "xgs_line_raw_cost": 0.0,
            "xgs_line_normalised_cost": 0.0
        })

# === STEP 2: Invoice-Level Aggregation ===


def estimate_total_freight(df: pd.DataFrame, historical_cost_col="freight_per_invoice") -> pd.DataFrame:
    """
    Estimate freight at line level, aggregate at invoice level including historical costs, raw and normalised XGS costs.

    Returns:
    - updated DataFrame with:
      - xgs_line_raw_cost
      - xgs_line_normalised_cost
      - total_invoice_historical_cost
      - total_invoice_xgs_raw_cost
      - total_invoice_xgs_normalised_cost
    """
    logging.info("🔧 Starting freight estimation...")

    df = df.copy()

    # Step 1: Line-level Freight Estimation
    freight_estimates = df.apply(estimate_line_invoice_freight_cost, axis=1)
    df = pd.concat([df, freight_estimates], axis=1)

    logging.info(
        f"✅ Line-level freight estimation complete for {df['invoice_id'].nunique()} invoices.")

    # Step 2: Invoice-level Summation
    logging.info("🔧 Summing freight costs at invoice level...")

    invoice_totals = df.groupby('invoice_id').agg({
        'xgs_line_raw_cost': 'sum',
        'xgs_line_normalised_cost': 'sum',
        historical_cost_col: 'first'  # Assume historical invoice freight is uniform
    }).reset_index()

    invoice_totals.rename(columns={
        'xgs_line_raw_cost': 'total_invoice_xgs_raw_cost',
        'xgs_line_normalised_cost': 'total_invoice_xgs_normalised_cost',
        historical_cost_col: 'total_invoice_historical_cost'
    }, inplace=True)

    df = df.merge(invoice_totals, on='invoice_id', how='left')

    logging.info("✅ Invoice-level freight aggregation complete.")

    return df
