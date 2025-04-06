from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import io
import os
from datetime import datetime

app = FastAPI()

# --------------------------
# Freight Class Breakpoints
# --------------------------
class_breakpoints = [
    ("L5C", 0, 499), ("5C", 500, 999), ("1M", 1000, 1999),
    ("2M", 2000, 2999), ("3M", 3000, 4999), ("5M", 5000, 9999),
    ("10M", 10000, 19999), ("20M", 20000, 29999), ("30M", 30000, 39999),
    ("40M", 40000, float("inf")),
]

# --------------------------
# Freight Rates & Discounts
# --------------------------
rates = {
    "SQYD": {
        "Texas": {
            "Carpet": {
                "L5C": 0.45, "5C": 0.42, "1M": 0.41, "2M": 0.40, "3M": 0.39,
                "5M": 0.38, "10M": 0.37, "20M": 0.36, "30M": 0.35, "40M": 0.34
            },
            "Carpet Tile": {
                "L5C": 0.75, "5C": 0.72, "1M": 0.70, "2M": 0.68, "3M": 0.66,
                "5M": 0.64, "10M": 0.62, "20M": 0.60, "30M": 0.58, "40M": 0.56
            }
        },
        "Florida": {
            "Carpet": {
                "L5C": 0.50, "5C": 0.47, "1M": 0.45, "2M": 0.44, "3M": 0.43,
                "5M": 0.42, "10M": 0.41, "20M": 0.40, "30M": 0.39, "40M": 0.38
            },
            "Carpet Tile": {
                "L5C": 0.78, "5C": 0.74, "1M": 0.71, "2M": 0.69, "3M": 0.67,
                "5M": 0.65, "10M": 0.63, "20M": 0.61, "30M": 0.59, "40M": 0.57
            }
        }
    },
    "CWT": {
        "Texas": {
            60: {
                "L5C": 20.0, "5C": 18.5, "1M": 14.2, "2M": 13.0, "3M": 12.0,
                "5M": 11.0, "10M": 10.0, "20M": 9.5, "30M": 9.0, "40M": 8.5
            },
            70: {
                "L5C": 23.0, "5C": 21.0, "1M": 16.5, "2M": 15.2, "3M": 14.0,
                "5M": 13.0, "10M": 12.0, "20M": 11.0, "30M": 10.5, "40M": 10.0
            }
        },
        "Florida": {
            60: {
                "L5C": 21.0, "5C": 19.0, "1M": 15.0, "2M": 14.0, "3M": 13.0,
                "5M": 12.0, "10M": 11.0, "20M": 10.5, "30M": 10.0, "40M": 9.5
            },
            70: {
                "L5C": 24.0, "5C": 22.0, "1M": 17.0, "2M": 16.0, "3M": 15.0,
                "5M": 14.0, "10M": 13.0, "20M": 12.0, "30M": 11.0, "40M": 10.5
            }
        }
    }
}

discounts = {
    "SQYD": {"Texas": 0.78, "Florida": 0.76},
    "CWT": {"Texas": 0.77, "Florida": 0.74}
}

# --------------------------
# Helpers
# --------------------------


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
    location = location.title()

    if uom == "SQFT":
        quantity = sqft_to_sqyd(quantity)
        uom = "SQYD"

    freight_class = get_priority_class(quantity)

    if uom == "SQYD":
        if not product_type:
            raise ValueError(
                "Product type must be provided for SQYD rate calculation.")
        product_type = product_type.title()
        rate = rates["SQYD"][location][product_type][freight_class]
        discount = discounts["SQYD"][location]
    elif uom == "CWT":
        if cwt_class is None:
            raise ValueError(
                "CWT class must be provided for CWT rate calculation.")
        rate = rates["CWT"][location][cwt_class][freight_class]
        discount = discounts["CWT"][location]
    else:
        raise ValueError(f"Unsupported unit of measure '{uom}'")

    cost = round(rate * discount * quantity, 2)
    return cost, freight_class, rate, discount

# --------------------------
# Models
# --------------------------


class FreightRequest(BaseModel):
    uom: str
    quantity: float
    location: str
    product_type: Optional[str] = None
    cwt_class: Optional[int] = None

# --------------------------
# Endpoints
# --------------------------


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
        required = ["PURCH UOM", "PO PURCH QTY", "SHIP TO ZIP"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return {"error": f"Missing columns: {', '.join(missing)}"}

        df["LOCATION"] = df.get("LOCATION", "Texas")

        def safe_estimate(row):
            try:
                cost, freight_class, rate, discount = estimate_freight_cost(
                    uom=row["PURCH UOM"],
                    quantity=row["PO PURCH QTY"],
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

        # Save results to downloads/
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
