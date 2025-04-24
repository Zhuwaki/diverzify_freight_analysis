# This module will contain only the loader utilities.
import pandas as pd
import logging
from typing import Dict

XGS_RATE_DISCOUNT = 0.06

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
                continue

            rate = row[col]
            try:
                if pd.notna(rate):
                    rate_table[site][unit][commodity][col.upper()] = (
                        float(
                            rate) / (1 + XGS_RATE_DISCOUNT) if apply_discount else float(rate)
                    )
            except ValueError:
                logging.warning(
                    f"⚠️ Skipped non-numeric rate '{rate}' in column '{col}' for {site}/{unit}/{commodity}"
                )

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
