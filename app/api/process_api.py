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
    add_freight_per_invoice
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


conversion_path = "app/conversion_table_standardized.csv"  # ✅ adjust if needed


@router.post("/process")
async def process_uploaded_file(file: UploadFile = File(...)):
    try:
        df = safe_read_uploaded_file(file)  # ✅ safe reading

        cleaned_df = data_cleaning(df)
        cleaned_df = uom_cleaning(cleaned_df)
        cleaned_df = flag_fully_converted_invoices(cleaned_df, conversion_path)
        enriched_df = enrich_invoice_flags(cleaned_df)
        enriched_df = add_freight_per_invoice(enriched_df)

        # Save intermediate file
        os.makedirs("downloads", exist_ok=True)
        temp_path = os.path.join("downloads", "intermediate_cleaned.csv")
        enriched_df.to_csv(temp_path, index=False)

        # Send to freight API
        with open(temp_path, "rb") as f:
            response = requests.post(
                "http://localhost:8000/batch", files={"file": f})

        if response.status_code != 200:
            return {"error": "Freight API failed", "detail": response.text}

        return response.json()

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
