
# === Global Toggles ===
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from utils.loaders import load_rate_table_from_csv, load_conversion_table


logging.basicConfig(level=logging.INFO)

RATES_CSV_PATH = "data/input/freight_model/freight_rates_operating.csv"
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
    "CWT": {"SPT": 1, "SPW": 1, "SPJ": 1, "DIT": 1, "SPN": 1, 'SPCP': 1, 'SPHU': 1, 'PVF': 1, 'SPTM': 1, 'CTS': 1, 'FSU': 1, },
    "SQFT": {"SPT": 1, "SPW": 1, "SPJ": 1, "DIT": 1, "SPN": 1, "SPCP": 1, "SPHU": 1, "PVF": 1, "SPTM": 1, 'CTS': 1, 'FSU': 1, },
    "SQYD": {"SPT": 1, "SPW": 1, "SPJ": 1, "DIT": 1, "SPN": 1, "SPCP": 1, "SPHU": 1, "PVF": 1, "SPTM": 1, 'CTS': 1, 'FSU': 1, },
}

# === Minimum freight charge per site ===
minimum_charges = {
    "SPT": {
        '1CBL': 60.64,
        '1VNL': 95.35,
        '1CPT': 60.64
    },
    "SPW": {
        '1CBL': 60.64,
        '1VNL': 95.35,
        '1CPT': 60.64
    },
    "SPJ": {
        '1CBL': 55.13,
        '1VNL': 95.35,
        '1CPT': 55.13
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

    "CTS": {
        '1CBL': 79.38,
        '1VNL': 148.31,
        '1CPT': 79.38
    },
    "FSU": {
        '1CBL': 123.04,
        '1VNL': 95.35,
        '1CPT': 123.04
    },
}
# === Loaders ===


rates = load_rate_table_from_csv(RATES_CSV_PATH)
conversion_lookup = load_conversion_table(CONVERSION_CSV_PATH)
print("✅ Loaded commodity groups from conversion table:", set(
    [v["commodity_group"] for v in conversion_lookup.values()]))


# === Utility Functions ===


def get_freight_class(quantity: float) -> str:
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
        raise ValueError(
            f"Conversion code '{conversion_code}' not found in lookup.")

    entry = conversion_lookup[code]
    normalized_uom = normalize_uom(entry["uom"])
    lbs_per_uom = entry["lbs_per_uom"]
    commodity_group = entry["commodity_group"]
    uom_used = entry["uom"]

    lbs = quantity * lbs_per_uom

    return lbs, normalized_uom, commodity_group, lbs_per_uom, uom_used


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


# Renamed and streamlined standardization function only
# ✅ Updated: Ensure 1CBL/1CPT rows using AREA method are processed without conversion lookup requirement

def standardize_commodity(
    quantity: float,
    inv_uom: str,
    commodity_group: str,
    conversion_code: str,
    site: str
) -> Dict:
    print(f"🧪 Estimating standardized quantity only...")

    def normalize_uom(uom: str) -> str:
        if not isinstance(uom, str):
            return ""
        uom = uom.strip().upper()
        return "SQFT" if uom == "SF" else "SQYD" if uom == "SY" else uom

    def sqft_to_sqyd(sqft: float) -> float:
        return sqft / 9

    # Initialize output structure
    output = {
        "standard_quantity": None,
        "standard_uom": None,
        "lbs_per_uom": None,
        "standardization_error": None
    }

    # Validate inputs
    if not isinstance(commodity_group, str) or not commodity_group.strip():
        output["standardization_error"] = f"Invalid commodity group '{commodity_group}' (must be a non-empty string)"
        return output

    if not isinstance(site, str) or not site.strip():
        output["standardization_error"] = f"Invalid site '{site}' (must be a non-empty string)"
        return output

    site = site.strip().upper()
    group = commodity_group.strip().upper()
    uom = normalize_uom(inv_uom)

    method = "CWT" if group == "1VNL" else "AREA" if group in [
        "1CBL", "1CPT"] else "N/A"

    if method == "AREA":
        if uom == "SQFT":
            std_qty = sqft_to_sqyd(quantity)
            std_uom = "SQYD"
        elif uom == "SQYD":
            std_qty = quantity
            std_uom = "SQYD"
        else:
            output["standardization_error"] = f"Unsupported UOM '{uom}' for AREA-based group"
            return output

        output.update({
            "standard_quantity": round(std_qty, 2),
            "standard_uom": std_uom,
            "lbs_per_uom": None,
            "standardization_error": "Standardization successful"
        })
        return output

    elif method == "CWT":
        try:
            lbs, input_uom, group, lbs_per_uom, uom_used = convert_area_to_weight(
                quantity, conversion_code)
        except Exception as e:
            output["standardization_error"] = f"Conversion failed: {e}"
            return output

        output.update({
            "standard_quantity": round(lbs, 2),
            "standard_uom": "LBS",
            "lbs_per_uom": lbs_per_uom,
            "standardization_error": "Standardization successful"
        })
        return output

    else:
        output["standardization_error"] = f"Unsupported commodity group '{commodity_group}'"
        return output


def classify_shipment_by_uom(qty: float, uom: str) -> str:
    print(f"🧪 Classifying shipment: qty={qty}, uom={uom}")
    if uom == 'LBS':
        return 'FTL' if qty > 19999 else 'LTL'
    elif uom == 'SQYD':
        return 'FTL' if qty > 2200 else 'LTL'
    else:
        return 'Unknown'
