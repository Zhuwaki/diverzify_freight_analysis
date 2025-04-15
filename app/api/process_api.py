import logging
import os
import io
import requests
import pandas as pd
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from utils.project_analysis_utils import (
    data_cleaning,
    enrich_invoice_flags,
    uom_cleaning,
    flag_fully_converted_invoices,
    add_freight_per_invoice,
    filter_valid_invoices
)

router = APIRouter()


def safe_read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    contents = file.file.read()  # ✅ this reads ONCE
    file.file.seek(0)            # ✅ reset pointer just in case

    filename = file.filename.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(contents), encoding="latin1")
    elif filename.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(contents), engine="openpyxl")
    elif filename.endswith(".xls"):
        return pd.read_excel(io.BytesIO(contents), engine="xlrd")
    else:
        raise ValueError("Unsupported file format")


# ✅ adjust if needed
conversion_path = "data/input/freight_model/conversion_table_standardized.csv"


@router.post("/process")
async def process_uploaded_file(file: UploadFile = File(...)):
    try:
        logging.info(
            "📥 /process called — reading uploaded file: %s", file.filename)

        df = safe_read_uploaded_file(file)
        logging.info("✅ File loaded — %s rows", df.shape[0])

        cleaned_df = data_cleaning(df)
        logging.info("🧹 Cleaning complete")

        cleaned_df = uom_cleaning(cleaned_df)
        logging.info("📏 UOM fix complete")

        cleaned_df = flag_fully_converted_invoices(cleaned_df, conversion_path)
        logging.info("🔁 Conversion flagging done")

        enriched_df = enrich_invoice_flags(cleaned_df)
        logging.info("📦 Enrichment complete")

        enriched_df = add_freight_per_invoice(enriched_df)
        logging.info("🚚 Freight per invoice added")

        filtered_df = filter_valid_invoices(enriched_df)
        logging.info("🚚 Data filtered")

        # Save intermediate file
        os.makedirs("downloads", exist_ok=True)
        temp_path = os.path.join("downloads", "intermediate_cleaned.csv")
        filtered_df.to_csv(temp_path, index=False)
        logging.info("💾 Intermediate file saved: %s", temp_path)

        # Send to freight API
        logging.info("📡 Sending file to /batch API")
        with open(temp_path, "rb") as f:
            logging.info("📦 File size (bytes): %s", os.path.getsize(temp_path))
            response = requests.post(
                "http://localhost:8000/batch", files={"file": f})

        if response.status_code != 200:
            logging.error("❌ /batch API failed: %s", response.text)
            return {"error": "Freight API failed", "detail": response.text}

        logging.info("✅ /batch API completed successfully")
        return response.json()

    except Exception as e:
        logging.exception("❌ Error in /process route")
        return JSONResponse(status_code=500, content={"error": str(e)})
