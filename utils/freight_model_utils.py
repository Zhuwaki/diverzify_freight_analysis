
# === Global Toggles ===
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
APPLY_XGS_DISCOUNT = False     # Toggle 6% discount from XGS rates (model)
APPLY_MARKET_DISCOUNT = False  # Toggle 30% discount from freight_price

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


# === Adjustable Rate Reduction Factors ===
MARKET_RATE_DISCOUNT = 0.30  # 30% off market freight price
XGS_RATE_DISCOUNT = 0.06     # 6% off XGS rates from the freight table


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
                            rate) * (1 - XGS_RATE_DISCOUNT) if APPLY_XGS_DISCOUNT else float(rate)
                    )

            except ValueError:
                logging.warning(
                    f"⚠️ Skipped non-numeric rate '{rate}' in column '{col}' for {site}/{unit}/{commodity}")

    logging.info("✅ Rate table loaded successfully.")
    return rate_table


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
    for class_name, min_q, max_q in class_breakpoints:
        if min_q <= quantity <= max_q:
            return class_name
    raise ValueError("Quantity is out of range.")


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
        # NEW RULE: Never calculate area for 1VNL
    if commodity_group.upper() == "1VNL":
        return "Not applicable", None, None, None

    if site not in rates or uom not in rates[site] or commodity_group not in rates[site][uom]:
        logging.info(
            f"ℹ️ Area pricing not available for {site} / {uom} / {commodity_group}")
        return "Not applicable", None, None, None

    try:
        freight_class = get_priority_class(quantity)
    except Exception as e:
        return f"Freight class error: {e}", None, None, None

    rate = rates[site][uom][commodity_group].get(freight_class)
    if rate is None:
        return "Missing class column", freight_class, None, None

    discount = discounts.get(uom, {}).get(site, 1)
    raw_cost = round(rate * discount * quantity, 2)
    # 👇 Fetch minimum charge by site and commodity group
    min_charge = minimum_charges.get(
        site, {}).get(commodity_group, 0)
    cost = round(max(raw_cost, min_charge), 2)

    return cost, freight_class, rate, discount


def estimate_dual_freight_cost(quantity: float, conversion_code: str, site: str) -> Dict:
    logging.info(f"📊 Calculating the dual freight costs")
    site = site.upper()
    min_charge = minimum_charges.get(site, 0)  # default to 0 if not defined
    try:
        lbs, original_uom, commodity_group = convert_area_to_weight(
            quantity, conversion_code)
    except Exception as e:
        return {"error": f"Conversion failed: {str(e)}"}

    # 👇 Fetch min_charge by site and commodity_group
    min_charge = minimum_charges.get(site, {}).get(commodity_group, 0)

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
            cwt_discount = discounts.get("CWT", {}).get(site, 1)
            raw_cost = cwt_rate * cwt_discount * cwt_quantity
            cwt_cost = round(max(raw_cost, min_charge), 2)
            cwt_min_applied = raw_cost < min_charge
    else:
        cwt_cost = "Not applicable"
        cwt_rate = None
        cwt_discount = None
        cwt_quantity = None
        freight_class = None
        cwt_min_applied = False

    est_sqyd = sqft_to_sqyd(quantity) if original_uom == "SQFT" else quantity

    if commodity_group in ["1CPT", "1CBL"]:
        area_cost, area_freight_class, area_rate, area_discount = estimate_area_based_cost(
            quantity, site, commodity_group, original_uom
        )
    else:
        area_cost, area_freight_class, area_rate, area_discount = "Not applicable", None, None, None

    if isinstance(area_cost, (int, float)):
        raw_area_cost = area_cost
        area_min_applied = raw_area_cost < min_charge
        area_cost = round(max(raw_area_cost, min_charge), 2)  # ← Apply minimum
    else:
        area_min_applied = False

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
        "original_uom": original_uom,
        "converted_sqyd": round(est_sqyd, 2),
        "freight_class_area": safe(area_freight_class),
        "rate_area": safe(area_rate, 0),
        "discount_area": safe(area_discount, 0),
        "estimated_area_cost": safe(area_cost, 0),
        "area_min_applied": bool(area_min_applied),
        "area_uom_used": "SQYD" if original_uom in ["SQFT", "SQYD"] else "N/A",
        "est_pricing_basis": pricing_basis,
        "min_rule_applied": bool(min_rule_applied),
        "raw_area_cost": safe(raw_area_cost, 0),
        "raw_cwt_cost": safe(raw_cwt_cost, 0),
        "freight_class_cwt": safe(freight_class),  # NEW
        "uom": safe(original_uom),                 # NEW
        "est_cwt_min_applied": bool(cwt_min_applied),  # NEW
        "est_area_min_applied": bool(area_min_applied),  # NEW
        "est_min_rule_applied": bool(min_rule_applied),  # NEW,
        'sqyd': est_sqyd,  # NEW
    }


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
