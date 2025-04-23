
# === Global Toggles ===
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

APPLY_XGS_DISCOUNT = True     # Toggle 6% discount from XGS rates (model)
APPLY_MARKET_DISCOUNT = True  # Toggle 30% discount from freight_price
APPLY_MINIMUM_CHARGES = True  # Toggle minimum charges (model)

# === Adjustable Discount Rates ===
XGS_RATE_DISCOUNT = 0.06
MARKET_RATE_DISCOUNT = 0.30


logging.basicConfig(level=logging.INFO)

RATES_CSV_PATH = "data/input/freight_model/freight_rates_updated.csv"
CONVERSION_CSV_PATH = "data/input/freight_model/conversion_table_standardized.csv"

# === Constants ===
class_breakpoints = [
    ("L5C", 0, 499), ("5C", 500, 999), ("1M", 1000, 1999),
    ("2M", 2000, 2999), ("3M", 3000, 4999), ("5M", 5000, 9999),
    ("10M", 10000, 19999), ("20M", 20000, 29999),
    ("30M", 30000, 39999), ("40M", 40000, float("inf")),
]

# === Discount structure (can be externalized later) ===
discounts = {
    "CWT": {"SPT": 1, "SPW": 1, "SPJ": 1, "DIT": 1, "SPN": 1, 'SPCP': 1, 'KUS': 1, 'SPHU': 1, 'PVF': 1, 'SPTM': 1},
    "SQFT": {"SPT": 1, "SPW": 1, "SPJ": 1, "DIT": 1, "SPN": 1, "SPCP": 1, "KUS": 1, "SPHU": 1, "PVF": 1, "SPTM": 1},
    "SQYD": {"SPT": 1, "SPW": 1, "SPJ": 1, "DIT": 1, "SPN": 1, "SPCP": 1, "KUS": 1, "SPHU": 1, "PVF": 1, "SPTM": 1},
}

# === Minimum freight charge per site ===
minimum_charges = {
    "SPT": {
        '1CBL': 0,
        '1VNL': 0,
        '1CPT': 0
    },
    "SPW": {
        '1CBL': 0,
        '1VNL': 0,
        '1CPT': 0
    },
    "SPJ": {
        '1CBL': 0,
        '1VNL': 0,
        '1CPT': 0
    },
    "DIT": {
        '1CBL': 52.59,
        '1VNL': 115.21,
        '1CPT': 52.59
    },
    "SPN": {
        '1CBL': 49.19,
        '1VNL': 95.35,
        '1CPT': 49.19
    },
    "SPCP": {
        '1CBL': 62.45,
        '1VNL': 108.59,
        '1CPT': 62.45
    },
    'SPHU': {
        '1CBL': 62.45,
        '1VNL': 101.96,
        '1CPT': 62.45
    },
    'SPTM': {
        '1CBL': 84.89,
        '1VNL': 135.07,
        '1CPT': 84.89
    },
    'PVF': {
        '1CBL': 79.38,
        '1VNL': 108.59,
        '1CPT': 79.38
    },
    "KUS": {
        '1CBL': 52.59,
        '1VNL': 115.21,
        '1CPT': 52.59
    }
}
# === Loaders ===


def load_rate_table_from_csv(filepath: str) -> Dict:
    df = pd.read_csv(filepath)
    df.columns = [col.strip().lower() for col in df.columns]

    # Define valid freight class columns from breakpoints
    valid_class_cols = [c[0].lower() for c in class_breakpoints]

    # Debug available columns
    logging.info(f"📊 Columns in freight rate table: {df.columns.tolist()}")

    required_cols = ["siteid", "unit", "commodity_group"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in rate table.")

    rate_table = {}

    for _, row in df.iterrows():
        site = row["siteid"].strip().upper()
        unit = row["unit"].strip().upper()
        commodity = str(row["commodity_group"]).strip().upper()

        rate_table.setdefault(site, {}).setdefault(
            unit, {}).setdefault(commodity, {})

        for col in df.columns:
            if col not in valid_class_cols:
                continue  # Skip anything that's not a freight class

            rate = row[col]
            try:
                if pd.notna(rate):
                    rate_table[site][unit][commodity][col.upper()] = (
                        float(
                            rate) / (1 + XGS_RATE_DISCOUNT) if APPLY_XGS_DISCOUNT else float(rate)
                    )

            except ValueError:
                logging.warning(
                    f"⚠️ Skipped non-numeric rate '{rate}' in column '{col}' for {site}/{unit}/{commodity}")

    logging.info("✅ Rate table loaded successfully.")
    print(rate_table)
    return rate_table

# This is a lookup table for conversion codes to commodity groups and units of measure


def load_conversion_table(filepath: str) -> Dict:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df["conversion_code"] = df["conversion_code"].str.strip().str.upper()
    logging.info("✅ Conversion table loaded successfully.")
    return {
        row["conversion_code"]: {
            "commodity_group": row["commodity_group"].strip().upper(),
            "uom": row["uom"].strip().upper(),
            "lbs_per_uom": row["lbs_per_uom"]
        }
        for _, row in df.iterrows()
    }


rates = load_rate_table_from_csv(RATES_CSV_PATH)
conversion_lookup = load_conversion_table(CONVERSION_CSV_PATH)

# === Utility Functions ===


def get_priority_class(quantity: float) -> str:
    if not isinstance(quantity, (int, float)) or pd.isna(quantity) or quantity < 0:
        raise ValueError(
            f"❌ Invalid quantity passed to freight class logic: {quantity}")

    quantity = round(quantity)
    for class_name, min_q, max_q in class_breakpoints:
        if min_q <= quantity <= max_q:
            return class_name

    raise ValueError(f"Quantity is out of range: {quantity}")


def sqft_to_sqyd(sqft: float) -> float:
    return sqft / 9


def normalize_uom(uom: str) -> str:
    uom = uom.strip().upper()
    return "SQFT" if uom == "SF" else "SQYD" if uom == "SY" else uom


def convert_area_to_weight(quantity: float, conversion_code: str):
    code = conversion_code.strip().upper()
    if code not in conversion_lookup:
        raise ValueError(f"Conversion code '{conversion_code}' not found.")
    entry = conversion_lookup[code]
    normalized_uom = normalize_uom(entry["uom"])
    lbs = quantity * entry["lbs_per_uom"]
    # (lbs, uom, commodity_group) tuple
    return lbs, normalized_uom, entry["commodity_group"]


def get_freight_rate(site: str, unit: str, commodity_group: str, freight_class: str) -> Tuple[Optional[float], Optional[str]]:
    site, unit, commodity_group, freight_class = site.upper(
    ), unit.upper(), commodity_group.upper(), freight_class.upper()
    try:
        if site not in rates:
            return None, f"Site '{site}' not found in rates"
        if unit not in rates[site]:
            return None, f"Unit '{unit}' not available at site '{site}'"
        if commodity_group not in rates[site][unit]:
            return None, f"Commodity group '{commodity_group}' not found under {site}/{unit}"

        rate = rates[site][unit][commodity_group].get(freight_class)
        if rate is None:
            available = list(rates[site][unit][commodity_group].keys())
            return None, f"Class '{freight_class}' not in {site}/{unit}/{commodity_group}. Available: {available}"
        return rate, None
    except Exception as e:
        return None, f"Rate lookup error: {str(e)}"


def estimate_area_based_cost(quantity: float, site: str, commodity_group: str, uom: str):
    uom = normalize_uom(uom)
    if uom == "SQFT":
        quantity = sqft_to_sqyd(quantity)
        uom = "SQYD"

    if commodity_group.upper() == "1VNL":
        return "Not applicable", None, None, None

    if site not in rates or uom not in rates[site] or commodity_group not in rates[site][uom]:
        logging.info(
            f"ℹ️ Area pricing not available for {site} / {uom} / {commodity_group}")
        return "Not applicable", None, None, None

    return calculate_freight_cost(quantity, site, uom, commodity_group, apply_minimum=APPLY_MINIMUM_CHARGES)


def estimate_freight_cost(
    quantity: float,
    inv_uom: str,
    commodity_group: str,
    conversion_code: str,
    site: str
) -> Dict:
    """
    Standardized freight estimation function with clear unit context:
    - Identifies method (CWT or AREA)
    - Normalizes quantity
    - Classifies freight class
    - Fetches rate and applies discount
    - Applies minimum charge logic
    - Returns rate unit meaning for interpretation
    """

    site = site.upper()
    group = commodity_group.upper()
    uom = normalize_uom(inv_uom)

    # 1. Determine method and default rate unit
    method = "CWT" if group == "1VNL" else "AREA" if group in [
        "1CBL", "1CPT"] else "N/A"
    rate_unit = "$/100 lbs" if method == "CWT" else "$/SQYD" if method == "AREA" else "N/A"

    # 2. Normalize quantity
    if method == "AREA":
        if uom == "SQFT":
            std_qty = sqft_to_sqyd(quantity)
            std_uom = "SQYD"
        elif uom == "SQYD":
            std_qty = quantity
            std_uom = "SQYD"
        else:
            return {"error": f"Unsupported UOM '{uom}' for AREA-based group"}
    elif method == "CWT":
        if uom == "SQFT":
            sqyd = sqft_to_sqyd(quantity)
        elif uom == "SQYD":
            sqyd = quantity
        else:
            return {"error": f"Unsupported UOM '{uom}' for CWT-based group"}
        try:
            lbs, _, _ = convert_area_to_weight(sqyd, conversion_code)
        except Exception as e:
            return {"error": f"Conversion failed: {e}"}
        std_qty = lbs
        std_uom = "LBS"
    else:
        return {"error": f"Unsupported commodity group '{group}'"}

    # 3. Classify freight class
    try:
        freight_class = get_priority_class(std_qty)
    except Exception as e:
        freight_class = None
        rate = None
        estimated_cost = f"Freight class error: {e}"
        min_applied = False
    else:
        rate_unit_key = "CWT" if method == "CWT" else "SQYD" if method == "AREA" else None
        rate, error = get_freight_rate(
            site, rate_unit_key, group, freight_class)

        if error:
            estimated_cost = error
            rate = None
            discount = None
            min_applied = False
        else:
            discount = discounts.get(std_uom, {}).get(site, 1)
            # Adjust rate unit: divide by 100 for CWT
            if method == "CWT":
                normalized_rate = rate / 100
            else:
                normalized_rate = rate

            raw_cost = normalized_rate * discount * std_qty

            min_charge = minimum_charges.get(site, {}).get(group, 0)
            rounded_raw_cost = round(raw_cost, 2)
            if APPLY_MINIMUM_CHARGES:
                estimated_cost = max(rounded_raw_cost, min_charge)
                min_applied = rounded_raw_cost < min_charge
            else:
                estimated_cost = rounded_raw_cost
                min_applied = False

    # 4. Classify as FTL or LTL
    shipment_type = classify_shipment_by_uom(std_qty, std_uom)

    return {
        "commodity_group": group,
        "method_used": method,
        "standard_quantity": round(std_qty, 2),
        "standard_uom": std_uom,
        "freight_class": freight_class,
        "rate": round(normalized_rate, 4),
        "rate_unit": "$/lb" if method == "CWT" else "$/SQYD",
        "discount": discount if rate else None,
        "raw_cost": rounded_raw_cost,
        "estimated_cost": estimated_cost,
        "min_applied": min_applied,
        "shipment_type": shipment_type
    }


def enrich_invoice_level_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates invoice-level total standard quantity and estimated market rate.
    Adds a 'market_estimated_rate' column based on:
        freight_per_invoice / total_standard_quantity

    Assumes:
    - 'invoice_id' exists
    - 'freight_per_invoice' is populated
    - 'est_standard_quantity' is numeric

    Returns:
    - df: enriched with 'market_estimated_rate'
    """
    import numpy as np

    # Ensure required columns exist
    required_cols = ["invoice_id",
                     "adjusted_freight_price", "est_standard_quantity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing values for aggregation
    valid_df = df.dropna(subset=required_cols)

    # Group and calculate invoice-level totals
    invoice_summary = valid_df.groupby("invoice_id").agg(
        total_standard_quantity=("est_standard_quantity", "sum"),
        adjusted_freight_price=("adjusted_freight_price", "first")
    ).reset_index()

    # Avoid divide-by-zero
    invoice_summary["market_estimated_rate"] = np.where(
        invoice_summary["total_standard_quantity"] > 0,
        invoice_summary["adjusted_freight_price"] /
        invoice_summary["total_standard_quantity"],
        np.nan
    )

    # Merge back to main dataframe
    df = df.merge(invoice_summary[[
                  "invoice_id", "market_estimated_rate"]], on="invoice_id", how="left")

    return df


def estimate_dual_freight_cost(quantity: float, inv_uom: str, commodity_group: str, conversion_code: str, site: str) -> Dict:

    logging.info(f"📊 Calculating the dual freight costs")
    site = site.upper()

    # 🔍 Standardize and classify before estimation
    analysis_uom, est_sqyd, lbs, shipment_type = standardize_quantity_and_classify(
        commodity_group,
        inv_uom,
        quantity,
        conversion_code
    )

    # 👇 Fetch min_charge by site and commodity_group
    min_charge = minimum_charges.get(site, {}).get(commodity_group, 0)

    # Calculate CWT cost if applicable - this is only applicable for 1VNL
    if commodity_group == "1VNL":
        cwt_quantity = lbs / 100

        freight_class = get_priority_class(lbs)
        cwt_rate, cwt_error = get_freight_rate(
            site, "CWT", commodity_group, freight_class
        )

        if cwt_error:
            cwt_cost = cwt_error
            cwt_discount = None
            cwt_min_applied = False
        else:
            cwt_cost, freight_class, cwt_rate, cwt_discount, cwt_min_applied, raw_cost = calculate_freight_cost(
                cwt_quantity, site, "CWT", commodity_group, apply_minimum=APPLY_MINIMUM_CHARGES
            )

    else:
        cwt_cost = "Not applicable"
        cwt_rate = None
        cwt_discount = None
        cwt_quantity = None
        freight_class = None
        cwt_min_applied = False

    # 1CPT and 1CBL are provided for by XGS
    if commodity_group in ["1CPT", "1CBL"]:
        area_cost, area_freight_class, area_rate, area_discount, area_min_applied, raw_area_cost = estimate_area_based_cost(
            quantity, site, commodity_group, inv_uom
        )

    else:
        area_cost, area_freight_class, area_rate, area_discount = "Not applicable", None, None, None
        area_min_applied = False
        raw_area_cost = None

    if commodity_group == "1VNL":
        pricing_basis = "CWT"
    elif commodity_group in ["1CBL", "1CPT"]:
        pricing_basis = "AREA"
    else:
        pricing_basis = "Not Applicable"

    # Rule-based labeling for where minimum was applied
    min_rule_applied = (
        (commodity_group == "1VNL" and area_min_applied) or
        (commodity_group == "1CBL" and cwt_min_applied)
    )

    raw_area_cost = None
    if isinstance(area_cost, (int, float)):
        raw_area_cost = area_cost
        area_min_applied = raw_area_cost < min_charge
        area_cost = round(max(raw_area_cost, min_charge), 2)
    else:
        area_min_applied = False

    raw_cwt_cost = None
    if not isinstance(cwt_cost, str):  # means it was calculated, not an error string
        raw_cwt_cost = round(raw_cost, 2)

    def safe(x, fallback="N/A"):
        if pd.isna(x) or x in [float("inf"), float("-inf")]:
            return fallback
        return x

    # 🔍 Determine shipment mode
    # 🔍 Determine shipment mode based on true weight or area
    if commodity_group == "1VNL":
        shipment_uom = "LBS"
        total_quantity = lbs  # ✅ Use the weight from convert_area_to_weight
    elif commodity_group in ["1CBL", "1CPT"]:
        shipment_uom = "SQYD"
        total_quantity = est_sqyd
    else:
        shipment_uom = None
        total_quantity = None

    # ✅ Classify as FTL or LTL using correct UOM and quantity
    shipment_type = classify_shipment_by_uom(total_quantity, shipment_uom)

    if commodity_group == "1VNL":
        xgs_real_rate = cwt_rate
    elif commodity_group in ["1CBL", "1CPT"]:
        xgs_real_rate = area_rate
    else:
        xgs_real_rate = None

    return {
        "commodity_group": commodity_group,
        "freight_class_lbs": safe(freight_class),  # freight_class,
        "lbs": safe(round(lbs, 2)) if isinstance(lbs, (int, float)) else 0,
        "cwt_quantity": safe(round(cwt_quantity, 2)) if isinstance(cwt_quantity, (int, float)) else 0,
        "weight_uom": "lbs",
        "rate_cwt": safe(cwt_rate, 0),
        "discount_cwt": safe(cwt_discount, 0),
        "estimated_cwt_cost": safe(cwt_cost, 0),
        "cwt_min_applied": bool(cwt_min_applied),
        "original_quantity": quantity,
        "original_uom": inv_uom,
        "converted_sqyd": round(est_sqyd, 2),
        "freight_class_area": safe(area_freight_class),
        "rate_area": safe(area_rate, 0),
        "discount_area": safe(area_discount, 0),
        "estimated_area_cost": safe(area_cost, 0),
        "area_min_applied": bool(area_min_applied),
        "area_uom_used": "SQYD" if inv_uom in ["SQFT", "SQYD"] else "N/A",
        "est_pricing_basis": pricing_basis,
        "min_rule_applied": bool(min_rule_applied),
        "raw_area_cost": safe(raw_area_cost, 0),
        "raw_cwt_cost": safe(raw_cwt_cost, 0),
        "freight_class_cwt": safe(freight_class),  # NEW
        "analysis_uom": safe(analysis_uom),                 # NEW
        "est_cwt_min_applied": bool(cwt_min_applied),  # NEW
        "est_area_min_applied": bool(area_min_applied),  # NEW
        "est_min_rule_applied": bool(min_rule_applied),  # NEW,
        'sqyd': est_sqyd,  # NEW
        'shipment_type': shipment_type,
        'xgs_real_rate': safe(xgs_real_rate, 0),  # NEW
    }


def calculate_freight_cost(quantity: float, site: str, unit: str, commodity_group: str, apply_minimum: bool = True):
    try:
        freight_class = get_priority_class(quantity)
    except Exception as e:
        return f"Freight class error: {e}", None, None, None, None

    rate, error = get_freight_rate(site, unit, commodity_group, freight_class)
    if error:
        return error, freight_class, None, None, None

    discount = discounts.get(unit, {}).get(site, 1)
    raw_cost = rate * discount * quantity
    min_charge = minimum_charges.get(site, {}).get(commodity_group, 0)

    if apply_minimum:
        cost = round(max(raw_cost, min_charge), 2)
        min_applied = raw_cost < min_charge
    else:
        cost = round(raw_cost, 2)
        min_applied = False

    if quantity is None or quantity <= 0:
        return f"Invalid quantity: {quantity}", None, None, None, None, None

    return cost, freight_class, rate, discount, min_applied, raw_cost


def classify_shipment_by_uom(qty: float, uom: str) -> str:
    print(f"🧪 Classifying shipment: qty={qty}, uom={uom}")
    if uom == 'LBS':
        return 'FTL' if qty >= 20000 else 'LTL'
    elif uom == 'SQYD':
        return 'FTL' if qty >= 2200 else 'LTL'
    else:
        return 'Unknown'


def apply_market_freight_discount(df: pd.DataFrame, column="freight_per_invoice") -> pd.DataFrame:
    """
    Adds a new column with the adjusted market freight rate.

    Parameters:
    - df: input DataFrame
    - column: name of column containing original freight values

    Returns:
    - DataFrame with new column 'adjusted_freight_price'
    """
    logging.info(
        f"✅ Applying market freight discount of {MARKET_RATE_DISCOUNT*100:.0f}% to '{column}'...")
    df['adjusted_freight_price'] = df[column] / (1 + MARKET_RATE_DISCOUNT)
    return df


def standardize_quantity_and_classify(
    commodity_group: str,
    inv_uom: str,
    invoiced_line_qty: float,
    conversion_code: str
) -> Tuple[str, float, float, str]:
    """
    Determines standard UOM, SQYD quantity, LBS, and final shipment type (FTL/LTL).
    """
    inv_uom = normalize_uom(inv_uom)
    commodity_group = commodity_group.upper()

    if commodity_group == "1VNL":
        # Always convert to SQYD first if SQFT
        if inv_uom == "SQFT":
            sqyd_qty = sqft_to_sqyd(invoiced_line_qty)
        elif inv_uom == "SQYD":
            sqyd_qty = invoiced_line_qty
        else:
            return "Unknown", None, None, "Unknown"

        # Convert to weight using conversion_code
        try:
            lbs, _, _ = convert_area_to_weight(sqyd_qty, conversion_code)
        except Exception as e:
            return "LBS", sqyd_qty, None, f"Error: {e}"

        shipment_type = classify_shipment_by_uom(lbs, "LBS")
        return "LBS", sqyd_qty, lbs, shipment_type

    elif commodity_group in ["1CBL", "1CPT"]:
        # Carpet-based products → use SQYD directly
        if inv_uom == "SQFT":
            sqyd_qty = sqft_to_sqyd(invoiced_line_qty)
        elif inv_uom == "SQYD":
            sqyd_qty = invoiced_line_qty
        else:
            return "Unknown", None, None, "Unknown"

        shipment_type = classify_shipment_by_uom(sqyd_qty, "SQYD")
        return "SQYD", sqyd_qty, None, shipment_type

    else:
        return "Unknown", None, None, "Unknown"
