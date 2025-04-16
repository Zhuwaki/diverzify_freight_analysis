from utils.project_analysis_utils import (
    data_cleaning,
    enrich_invoice_flags,
    uom_cleaning,
    flag_fully_converted_invoices,
    add_freight_per_invoice,
    filter_valid_invoices
)
import sys
import os
import io
import pandas as pd
from fastapi import UploadFile, File, APIRouter
from fastapi.responses import FileResponse
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


router = APIRouter()

print(sys.path)

conversion_csv_path = "data/input/freight_model/conversion_table_standardized.csv"


@router.post("/clean")
async def clean_raw_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin1")
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file format"}

        # ✅ Apply your processing pipeline
        cleaned_df = data_cleaning(df)
        uom_df = uom_cleaning(cleaned_df)
        df_converted = flag_fully_converted_invoices(
            uom_df, conversion_csv_path)
        enriched_df = enrich_invoice_flags(df_converted)
        freight_df = add_freight_per_invoice(enriched_df)
        filtered_df = filter_valid_invoices(freight_df)

        # ✅ Save cleaned output
        cleaned_filename = f"cleaned_{file.filename.replace(' ', '_')}"
        output_path = os.path.join("data/downloads", cleaned_filename)
        os.makedirs("data/downloads", exist_ok=True)
        filtered_df.to_csv(output_path, index=False)

        return FileResponse(output_path, media_type="text/csv", filename=cleaned_filename)

    except Exception as e:
        return {"error": str(e)}
