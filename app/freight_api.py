
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import io
import os
from datetime import datetime

app = FastAPI()

class_breakpoints = [
    ("L5C", 0, 499), ("5C", 500, 999), ("1M", 1000, 1999),
    ("2M", 2000, 2999), ("3M", 3000, 4999), ("5M", 5000, 9999),
    ("10M", 10000, 19999), ("20M", 20000, 29999), ("30M", 30000, 39999),
    ("40M", 40000, float("inf")),
]


def load_rate_table_from_csv(filepath: str):
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    rate_table = {}

    for _, row in df.iterrows():
        site = row["SiteID"].upper()
        unit = row["Unit"].upper()
        commodity = str(row["CommodityGroup"]).strip().upper()

        rate_table.setdefault(site, {}).setdefault(
            unit, {}).setdefault(commodity, {})

        for col in df.columns:
            if col in ["Site", "SiteID", "Unit", "CommodityGroup", "UnitClass", "FreightClass", "OldFreightClass", "CommodityDescription"]:
                continue
            rate = row[col]
            if pd.notna(rate):
                rate_table[site][unit][commodity][col] = float(rate)

    return rate_table


RATES_CSV_PATH = "freight_rates_updated.csv"
CONVERSION_CSV_PATH = "conversion_table_with_code.csv"

rates = load_rate_table_from_csv(RATES_CSV_PATH)
conversion_df = pd.read_csv(CONVERSION_CSV_PATH)
conversion_df["ConversionCode"] = conversion_df["ConversionCode"].str.strip(
).str.upper()

conversion_lookup = {
    row["ConversionCode"]: {
        "CommodityGroup": row["CommodityGroup"].strip().upper(),
        "UOM": row["UOM"].strip().upper(),
        "LbsPerUOM": row["LbsPerUOM"]
    }
    for _, row in conversion_df.iterrows()
}

discounts = {
    "CWT": {"SPT": 1, "SPW": 1, "SPJ": 1},
    "SQYD": {"SPT": 1, "SPW": 1, "SPJ": 1}
}


def get_priority_class(quantity: float) -> str:
    for class_name, min_q, max_q in class_breakpoints:
        if min_q <= quantity <= max_q:
            return class_name
    raise ValueError("Quantity is out of range.")


def sqft_to_sqyd(quantity: float) -> float:
    return quantity / 9


def convert_area_to_weight(quantity: float, conversion_code: str) -> (float, str, str):
    code = conversion_code.strip().upper()
    if code not in conversion_lookup:
        raise ValueError(f"Conversion code '{conversion_code}' not found.")
    entry = conversion_lookup[code]
    lbs = quantity * entry["LbsPerUOM"]
    return lbs, entry["UOM"], entry["CommodityGroup"]


def estimate_area_based_cost(quantity: float, site: str, commodity_group: str, uom: str):
    if uom == "SQFT":
        quantity = sqft_to_sqyd(quantity)
        uom = "SQYD"

    freight_class = get_priority_class(quantity)

    if site not in rates or uom not in rates[site] or commodity_group not in rates[site][uom]:
        return None, None, None, None
    rate_group = rates[site][uom][commodity_group]
    if freight_class not in rate_group:
        return None, None, None, None
    rate = rate_group[freight_class]
    discount = discounts.get(uom, {}).get(site, 1)
    cost = round(rate * discount * quantity, 2)
    return cost, freight_class, rate, discount


def estimate_dual_freight_cost(quantity: float, conversion_code: str, site: str):
    site = site.upper()
    lbs, original_uom, commodity_group = convert_area_to_weight(
        quantity, conversion_code)
    cwt_quantity = lbs / 100

    # CWT calculation
    cwt_freight_class = get_priority_class(cwt_quantity)
    cwt_rate = rates[site]["CWT"][commodity_group][cwt_freight_class]
    cwt_discount = discounts["CWT"].get(site, 1)
    cwt_cost = round(cwt_rate * cwt_discount * cwt_quantity, 2)

    # Area-based cost
    area_cost, area_freight_class, area_rate, area_discount = estimate_area_based_cost(
        quantity, site, commodity_group, original_uom)

    return {
        "estimated_cwt_cost": cwt_cost,
        "freight_class_cwt": cwt_freight_class,
        "rate_cwt": cwt_rate,
        "discount_cwt": cwt_discount,
        "estimated_area_cost": area_cost,
        "freight_class_area": area_freight_class,
        "rate_area": area_rate,
        "discount_area": area_discount,
        "commodity_group": commodity_group,
        "uom": original_uom
    }


@app.post("/estimate")
def estimate_dual(data: dict):
    try:
        result = estimate_dual_freight_cost(
            quantity=data["quantity"],
            conversion_code=data["conversion_code"],
            site=data["site"]
        )
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/batch")
async def estimate_dual_batch(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename.endswith((".xls", ".xlsx")):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file type. Please upload .csv or .xlsx"}

        df.columns = [col.strip().upper() for col in df.columns]
        required = ["SITEID", "QUANTITY", "CONVERSIONCODE"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return {"error": f"Missing columns: {', '.join(missing)}"}

        def safe_dual(row):
            try:
                print("[DEBUG] Processing row:", row.to_dict())
                result = estimate_dual_freight_cost(
                    quantity=row["QUANTITY"],
                    conversion_code=row["CONVERSIONCODE"],
                    site=row["SITEID"]
                )
                return pd.Series(result)
            except Exception as e:
                print("[ERROR] Failed processing row:", row.to_dict())
                print("[ERROR] Exception:", str(e))
                return pd.Series({
                    "estimated_cwt_cost": f"Error: {str(e)}",
                    "freight_class_cwt": "",
                    "rate_cwt": "",
                    "discount_cwt": "",
                    "estimated_area_cost": "",
                    "freight_class_area": "",
                    "rate_area": "",
                    "discount_area": "",
                    "commodity_group": "",
                    "uom": ""
                })

        results = df.apply(safe_dual, axis=1)
        final_df = pd.concat([df, results], axis=1)

        os.makedirs("downloads", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"freight_dual_results_{timestamp}.csv"
        final_path = os.path.join("downloads", output_file)
        final_df.to_csv(final_path, index=False)

        return {
            "filename": file.filename,
            "rows_processed": len(final_df),
            "preview": final_df.head(5).to_dict(orient="records"),
            "download_url": f"/download/{output_file}"
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/download/{filename}")
def download_result(filename: str):
    filepath = os.path.join("downloads", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='text/csv', filename=filename)
    return {"error": "File not found."}
