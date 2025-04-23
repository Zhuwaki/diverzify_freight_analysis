
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
SURCHARGE_DISCOUNT = 0.30


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
    # print(code)
    if code not in conversion_lookup:
        raise ValueError(f"Conversion code '{conversion_code}' not found.")
    entry = conversion_lookup[code]
    print(entry)
    normalized_uom = normalize_uom(entry["uom"])
    lbs = quantity * entry["lbs_per_uom"]
    # print('this', quantity, 'is now', lbs)
    # (lbs, uom, commodity_group) tuple
    return lbs, normalized_uom, entry["commodity_group"], entry["lbs_per_uom"], entry["uom"]


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


def estimate_freight_cost(
    quantity: float,
    inv_uom: str,
    commodity_group: str,
    conversion_code: str,
    site: str
) -> Dict:

    print(f"🧪 Estimating freight costs...")
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

    print(f"🧪 Define method ...")
    # 1. Determine method and default rate unit
    method = "CWT" if group == "1VNL" else "AREA" if group in [
        "1CBL", "1CPT"] else "N/A"
    print(f"🧪 Normalise UOM ...")
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
            lbs, input_uom, commodity_group, lbs_per_uom, uom_used = convert_area_to_weight(
                quantity, conversion_code)

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
    # shipment_type = classify_shipment_by_uom(std_qty, std_uom)

    if method == 'CWT':
        print("Processing......", freight_class, commodity_group, 'with input quantity of', quantity, 'and input UOM of', input_uom,
              'converted to', std_qty, std_uom,
              'using coversion code:', conversion_code,
              f'with conversion rate of {lbs_per_uom} lbs per {uom_used}')
    elif method == 'AREA':
        print("Processing......", freight_class,  commodity_group, 'with input quantity of', quantity, 'and input UOM of', inv_uom,
              'converted to', std_qty, std_uom,
              )

    return {
        "commodity_group": group,
        "method_used": method,
        "standard_quantity": round(std_qty, 2),
        "standard_uom": std_uom,
        # "freight_class": freight_class,
        # "shipment_type": shipment_type,
        # "xgs_rate": round(normalized_rate, 4),
        # "rate_unit": "$/lb" if method == "CWT" else "$/SQYD",
        # "discount": discount if rate else None,
        # "xgs_actual_cost": rounded_raw_cost,
        # "xgs_normalised_cost": estimated_cost,
        # "min_applied": min_applied,
    }


def adjust_market_rate_surcharge(df: pd.DataFrame, column="freight_per_invoice") -> pd.DataFrame:
    print(f"🧪 Adjusting for surcharge")
    """
    Adds a new column with the adjusted market freight rate.

    Parameters:
    - df: input DataFrame
    - column: name of column containing original freight values

    Returns:
    - DataFrame with new column 'adjusted_freight_price'
    """
    logging.info(
        f"✅ Applying market freight discount of {SURCHARGE_DISCOUNT*100:.0f}% to '{column}'...")
    df['est_market_freight_costs'] = df[column] / (1 + SURCHARGE_DISCOUNT)
    return df


def estimate_market_rates(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🧪 Estimating market rates")
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
                     "est_market_freight_costs", "est_standard_quantity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing values for aggregation
    valid_df = df.dropna(subset=required_cols)

    # Group and calculate invoice-level totals
    invoice_summary = valid_df.groupby("invoice_id").agg(
        total_standard_quantity=("est_standard_quantity", "sum"),
        est_market_freight_costs=("est_market_freight_costs", "first")
    ).reset_index()

    # Avoid divide-by-zero
    invoice_summary["est_market_rate"] = np.where(
        invoice_summary["total_standard_quantity"] > 0,
        invoice_summary["est_market_freight_costs"] /
        invoice_summary["total_standard_quantity"],
        np.nan
    )

    # Merge back to main dataframe
    df = df.merge(invoice_summary[[
                  "invoice_id", "est_market_rate"]], on="invoice_id", how="left")

    return df


def estimate_xgs_rates(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🧪 Estimating XGS rates")
    """
    Calculates invoice-level total standard quantity and two estimated XGS rates:
    - est_xgs_actual_rate = est_xgs_actual_cost / total_standard_quantity
    - est_xgs_normalised_rate = est_xgs_normalised_cost / total_standard_quantity

    Assumes:
    - 'invoice_id' exists
    - 'est_xgs_actual_cost' and 'est_xgs_normalised_cost' are populated
    - 'est_standard_quantity' is numeric

    Returns:
    - df: enriched with 'est_xgs_actual_rate' and 'est_xgs_normalised_rate'
    """
    import numpy as np

    required_cols = ["invoice_id",  "est_standard_quantity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing data for any of the required columns
    valid_df = df.dropna(subset=required_cols)

    # Group by invoice and compute total standard quantity and first cost values
    invoice_summary = valid_df.groupby("invoice_id").agg(
        total_standard_quantity=("est_standard_quantity", "sum"),
        est_xgs_actual_cost=("est_xgs_actual_cost", "mean"),
        est_xgs_normalised_cost=("est_xgs_normalised_cost", "mean")
    ).reset_index()

    # Calculate both rates
    invoice_summary["est_xgs_actual_rate"] = np.where(
        invoice_summary["total_standard_quantity"] > 0,
        invoice_summary["est_xgs_actual_cost"] /
        invoice_summary["total_standard_quantity"],
        np.nan
    )

    invoice_summary["est_xgs_normalised_rate"] = np.where(
        invoice_summary["total_standard_quantity"] > 0,
        invoice_summary["est_xgs_normalised_cost"] /
        invoice_summary["total_standard_quantity"],
        np.nan
    )

    # Merge rates back into original DataFrame
    df = df.merge(
        invoice_summary[["invoice_id", 'total_standard_quantity', "est_xgs_actual_rate",
                         "est_xgs_normalised_rate"]],
        on="invoice_id",
        how="left"
    )

    return df


def classify_shipment_by_uom(qty: float, uom: str) -> str:
    print(f"🧪 Classifying shipment: qty={qty}, uom={uom}")
    if uom == 'LBS':
        return 'FTL' if qty > 19999 else 'LTL'
    elif uom == 'SQYD':
        return 'FTL' if qty > 2200 else 'LTL'
    else:
        return 'Unknown'


def output_data(df):
    print(f"🧪 Cleaning columns")
    output_columns = [


        "site",
        "site_description",
        "supplier_no",
        "supplier_name",
        "invoice_id",
        'po_no',
        'ship_to_zip',
        "part_no",
        "part_description",
        "est_commodity_group",
        "new_commodity_description",
        "conversion_code",
        "invoiced_line_qty",
        "inv_uom",
        "freight_per_invoice",
        "est_method_used",
        "est_standard_quantity",
        "est_standard_uom",
        # "est_freight_class",
        #  "est_shipment_type",
        #  "est_xgs_rate",
        # "est_rate_unit",
        # "est_discount",
        # "est_xgs_actual_cost",
        # "est_xgs_normalised_cost",
        # "est_min_applied",
        "est_market_freight_costs",
        "est_market_rate",
        # "est_xgs_actual_rate",
        # "est_xgs_normalised_rate",
        'total_standard_quantity',
        "invoice_id",
        "invoice_freight_class",
        "invoice_rate",
        "invoice_rate_unit",
        "invoice_shipment_type"
    ]

    df = df[output_columns]

    return df


def calculate_invoice_quantity(df):
    """
    Calculates total standard quantity for each invoice.

    Assumes:
    - 'invoice_id', 'est_standard_quantity' exist

    Returns:
    - df: merged with total_standard_quantity
    """
    # Ensure required columns exist
    required_cols = ["invoice_id", "est_standard_quantity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Group by invoice_id and sum est_standard_quantity
    invoice_summary = df.groupby("invoice_id").agg(
        total_standard_quantity=("est_standard_quantity", "sum")
    ).reset_index()

    # Merge back to main dataframe
    df = df.merge(invoice_summary, on="invoice_id", how="left")

    return df


def classify_invoice_level_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines invoice-level freight class, rate, rate unit, and shipment type (LTL/FTL).

    Assumes:
    - 'invoice_id', 'total_standard_quantity', 'site', 'est_commodity_group', 'est_standard_uom' exist

    Returns:
    - df: merged with invoice-level freight_class, rate, rate_unit, and shipment_type
    """
    results = []

    for _, row in df.iterrows():
        try:
            site = row["site"]
            group = row["est_commodity_group"]
            total_qty = row["total_standard_quantity"]
            std_uom = row["est_standard_uom"]

            # Determine method and rate unit
            method = "CWT" if group == "1VNL" else "AREA"
            rate_unit = "CWT" if method == "CWT" else "SQYD"

            # Freight class based on total standardized quantity
            freight_class = get_priority_class(total_qty)

            # Rate lookup
            rate, error = get_freight_rate(
                site, rate_unit, group, freight_class)
            if error:
                raise ValueError(error)

            # Shipment classification
            shipment_type = classify_shipment_by_uom(
                total_qty, "LBS" if method == "CWT" else "SQYD")

        except Exception as e:
            freight_class = None
            rate = None
            rate_unit = None
            shipment_type = None

        results.append({
            "invoice_id": row["invoice_id"],
            "invoice_freight_class": freight_class,
            "invoice_rate": rate/100 if method == "CWT" else rate,
            "invoice_rate_unit": "$/lbs" if method == "CWT" else "$/SQYD",
            "invoice_shipment_type": shipment_type
        })

    return df.merge(pd.DataFrame(results), on="invoice_id", how="left")
