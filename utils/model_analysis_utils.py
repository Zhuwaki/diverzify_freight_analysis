import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Tuple, Dict

import logging
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
    "CWT": {"SPT": 1, "SPW": 1, "SPJ": 1},
    "SQFT": {"SPT": 1, "SPW": 1, "SPJ": 1},
    "SQYD": {"SPT": 1, "SPW": 1, "SPJ": 1}
}

# === Minimum freight charge per site ===
minimum_charges = {
    "SPT": 0,
    "SPW": 0,
    "SPJ": 0,
    "DIT": 95.35,
    "SPN": 49.19,
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
                    rate_table[site][unit][commodity][col.upper()
                                                      ] = float(rate)
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

    freight_class = get_priority_class(quantity)
    rate = rates[site][uom][commodity_group].get(freight_class)
    if rate is None:
        return "Missing class column", freight_class, None, None

    discount = discounts.get(uom, {}).get(site, 1)
    cost = round(rate * discount * quantity, 2)
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

    cwt_quantity = lbs / 100
    freight_class = get_priority_class(lbs)
    cwt_rate, cwt_error = get_freight_rate(
        site, "CWT", commodity_group, freight_class)

    if cwt_error:
        cwt_cost = cwt_error
        cwt_discount = None
        cwt_min_applied = False  # ← Add tracking for min logic
    else:
        cwt_discount = discounts.get("CWT", {}).get(site, 1)
        raw_cost = cwt_rate * cwt_discount * cwt_quantity
        cwt_cost = round(max(raw_cost, min_charge), 2)  # ← Apply minimum
        cwt_min_applied = raw_cost < min_charge

    est_sqyd = sqft_to_sqyd(quantity) if original_uom == "SQFT" else quantity
    area_cost, area_freight_class, area_rate, area_discount = estimate_area_based_cost(
        quantity, site, commodity_group, original_uom)

    if isinstance(area_cost, (int, float)):
        raw_area_cost = area_cost
        area_min_applied = raw_area_cost < min_charge
        area_cost = round(max(raw_area_cost, min_charge), 2)  # ← Apply minimum
    else:
        area_min_applied = False

    # Determine pricing basis based on which estimate succeeded
    if isinstance(cwt_cost, (int, float)) and isinstance(area_cost, str):
        pricing_basis = "CWT"
    elif isinstance(area_cost, (int, float)) and isinstance(cwt_cost, str):
        pricing_basis = "AREA"
    elif isinstance(cwt_cost, (int, float)) and isinstance(area_cost, (int, float)):
        pricing_basis = "CWT + AREA"
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
        "freight_class_lbs": freight_class,
        "lbs": round(lbs, 2),
        "cwt_quantity": round(cwt_quantity, 2),
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
