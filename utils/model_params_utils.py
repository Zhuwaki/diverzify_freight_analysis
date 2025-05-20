# This module will contain only the loader utilities.
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict


XGS_RATE_DISCOUNT = 0.06
XGS_FUEL_SURCHARGE = 0.3
XGS_LTL_REBATE = 0.1
STARNET_REBATE = 0.025

RATES_CSV_PATH = "data/input/freight_model/freight_rates_operating_multi.csv"

CONVERSION_CSV_PATH = "data/input/freight_model/conversion_table_standardized_15052025_v2.csv"

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


# === Loaders ===


# === Constants for loader use ===
class_breakpoints = [
    ("L5C", 0, 499), ("5C", 500, 999), ("1M", 1000, 1999),
    ("2M", 2000, 2999), ("3M", 3000, 4999), ("5M", 5000, 9999),
    ("10M", 10000, 19999), ("20M", 20000, 29999),
    ("30M", 30000, 39999), ("40M", 40000, float("inf")),
]

logging.basicConfig(level=logging.INFO)


def load_rate_table_from_csv(filepath: str, apply_discount: bool = True) -> Dict:
    df = pd.read_csv(filepath)
    df.columns = [col.strip().lower() for col in df.columns]

    valid_class_cols = [c[0].lower() for c in class_breakpoints]
    required_cols = ["site", "unit", "commodity_group"]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in rate table.")

    rate_table = {}

    for _, row in df.iterrows():
        site = row["site"].strip().upper()
        unit = row["unit"].strip().upper()
        commodity = str(row["commodity_group"]).strip().upper()

        rate_table.setdefault(site, {}).setdefault(
            unit, {}).setdefault(commodity, {})

        for col in df.columns:
            if col in valid_class_cols:
                rate = row[col]
                try:
                    if pd.notna(rate):
                        rate_table[site][unit][commodity][col.upper()
                                                          ] = float(rate)
                except ValueError:
                    logging.warning(
                        f"⚠️ Skipped non-numeric rate '{rate}' in column '{col}' for {site}/{unit}/{commodity}"
                    )

        # Load minimum_charge and ftl_flat_rate into metadata
        rate_table[site][unit][commodity]['__meta__'] = {
            'minimum_charge': row.get('minimum_charge', np.nan),
            'ftl_flat_rate': row.get('ftl_flat_rate', np.nan),
            'old_discount': row.get('old_discount', np.nan),
            'new_discount': row.get('new_discount', np.nan),
        }

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


def get_freight_rate(site: str, unit: str, commodity_group: str, freight_class: str) -> Tuple[Optional[float], Optional[str]]:

    rates = load_rate_table_from_csv(RATES_CSV_PATH)
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
        meta = rates[site][unit][commodity_group].get('__meta__', {})

        if rate is None:
            available = list(rates[site][unit][commodity_group].keys())
            return None, f"Class '{freight_class}' not in {site}/{unit}/{commodity_group}. Available: {available}"

        # ✅ New Vinyl-specific logic
        if commodity_group == '1VNL':
            old_disc = meta.get('old_discount')
            new_disc = meta.get('new_discount')
            if old_disc is not None and new_disc is not None:
                rate = (rate / (1 - old_disc)) * (1 - new_disc)

        # adjust rate for FSC and rebates
        inflation_rate = rate / (1 + XGS_RATE_DISCOUNT)
        fsc_rate = inflation_rate * (1 + XGS_FUEL_SURCHARGE)
        xgs_rebate = inflation_rate * (XGS_LTL_REBATE)
        interim_rate = fsc_rate - xgs_rebate
        star_net_rebate = (inflation_rate - xgs_rebate) * (STARNET_REBATE)
        final_rate = fsc_rate - xgs_rebate - star_net_rebate

        return {
            "base_rate": rate,
            "inflation_rate": inflation_rate,
            "fsc_rate": fsc_rate,
            "xgs_rebate": xgs_rebate,
            'fsc_xgs_rebate': interim_rate,
            "star_net_rebate": star_net_rebate,
            "final_rate": final_rate,
            "minimum_charge": meta.get("minimum_charge", None),
            "ftl_flat_rate": meta.get("ftl_flat_rate", None)
        }, None

    except Exception as e:
        return None, f"Rate lookup error: {str(e)}"


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


conversion_lookup = load_conversion_table(CONVERSION_CSV_PATH)
print("✅ Loaded commodity groups from conversion table:", set(
    [v["commodity_group"] for v in conversion_lookup.values()]))


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
