
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
        location = row["SiteID"].upper()
        unit = row["Unit"].upper()
        class_key = str(row["CommodityGroup"]).strip().title()

        rate_table.setdefault(unit, {}).setdefault(
            location, {}).setdefault(class_key, {})

        for col in df.columns:
            if col in ["Site", "SiteID", "Unit", "CommodityGroup"]:
                continue
            rate = row[col]
            if pd.notna(rate):
                rate_table[unit][location][class_key][col.strip()
                                                      ] = float(rate)

    return rate_table


RATES_CSV_PATH = "freight_rates.csv"
rates = load_rate_table_from_csv(RATES_CSV_PATH)

discounts = {
    "SQYD": {"SPT": 1, "SPW": 1, "SPJ": 1},
    "CWT": {"SPT": 1, "SPW": 1, "SPJ": 1}
}


def sqft_to_sqyd(sqft):
    return sqft / 9


def get_priority_class(quantity: float) -> str:
    for class_name, min_q, max_q in class_breakpoints:
        if min_q <= quantity <= max_q:
            return class_name
    raise ValueError("Quantity is out of range.")


def estimate_freight_cost(uom: str, quantity: float, location: str,
                          product_type: Optional[str] = None, cwt_class: Optional[int] = None):
    uom = uom.upper()
    location = location.upper()

    if uom == "SQFT":
        quantity = sqft_to_sqyd(quantity)
        uom = "SQYD"

    freight_class = get_priority_class(quantity)

    if uom == "SQYD":
        if not product_type:
            raise ValueError(
                "Product type must be provided for SQYD rate calculation.")
        product_type = product_type.title()
        rate = rates[uom][location][product_type][freight_class]
        discount = discounts[uom][location]
    elif uom == "CWT":
        if cwt_class is None:
            raise ValueError(
                "CWT class must be provided for CWT rate calculation.")
        rate = rates[uom][location][str(cwt_class)][freight_class]
        discount = discounts[uom][location]
    else:
        raise ValueError(f"Unsupported unit of measure '{uom}'")

    cost = round(rate * discount * quantity, 2)
    return cost, freight_class, rate, discount


class FreightRequest(BaseModel):
    uom: str
    quantity: float
    location: str
    product_type: Optional[str] = None
    cwt_class: Optional[int] = None


@app.post("/estimate")
def estimate_freight(data: FreightRequest):
    try:
        cost, freight_class, rate, discount = estimate_freight_cost(
            uom=data.uom,
            quantity=data.quantity,
            location=data.location,
            product_type=data.product_type,
            cwt_class=data.cwt_class
        )
        return {
            "estimated_cost": cost,
            "freight_class": freight_class,
            "rate": rate,
            "discount": discount
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/batch")
async def estimate_batch(file: UploadFile = File(...)):
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
        required = ["SITE", "PO INV QTY", "INV UOM",
                    "PART DESCRIPTION", "COMMODITY GROUP"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return {"error": f"Missing columns: {', '.join(missing)}"}

        df["LOCATION"] = df["SITE"]

        def safe_estimate(row):
            try:
                cost, freight_class, rate, discount = estimate_freight_cost(
                    uom=row["INV UOM"],
                    quantity=row["PO INV QTY"],
                    location=row["LOCATION"],
                    product_type=row["PART DESCRIPTION"] if pd.notna(
                        row.get("PART DESCRIPTION")) else None,
                    cwt_class=int(row["COMMODITY GROUP"]) if pd.notna(
                        row.get("COMMODITY GROUP")) else None
                )
                return pd.Series({
                    "ESTIMATED_FREIGHT_COST": cost,
                    "FREIGHT_CLASS": freight_class,
                    "RATE_USED": rate,
                    "DISCOUNT_USED": discount
                })
            except Exception as e:
                return pd.Series({
                    "ESTIMATED_FREIGHT_COST": f"Error: {str(e)}",
                    "FREIGHT_CLASS": "",
                    "RATE_USED": "",
                    "DISCOUNT_USED": ""
                })

        result_df = df.apply(safe_estimate, axis=1)
        final_df = pd.concat([df, result_df], axis=1)

        final_df.replace([float("inf"), float("-inf")], None, inplace=True)
        final_df.fillna("", inplace=True)

        os.makedirs("downloads", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"freight_results_{timestamp}.csv"
        result_path = os.path.join("downloads", result_filename)
        final_df.to_csv(result_path, index=False)

        return {
            "filename": file.filename,
            "rows_processed": len(final_df),
            "preview": final_df.head(5).to_dict(orient="records"),
            "download_url": f"/download/{result_filename}"
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/download/{filename}")
def download_result(filename: str):
    filepath = os.path.join("downloads", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='text/csv', filename=filename)
    return {"error": "File not found."}
