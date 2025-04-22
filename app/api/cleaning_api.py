from utils.data_cleaning_utils import (
    data_cleaning,
    enrich_invoice_flags,
    uom_cleaning,
    flag_fully_converted_invoices,
    add_freight_per_invoice,
    filter_valid_invoices,
)
import traceback
import orjson
import os
import io
import pandas as pd
from datetime import datetime
from fastapi import UploadFile, File, APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()
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

        df = data_cleaning(df)
        df = uom_cleaning(df)
        df = flag_fully_converted_invoices(df, conversion_csv_path)
        df = enrich_invoice_flags(df)
        df = add_freight_per_invoice(df)
        df = filter_valid_invoices(df)

        df = df.replace([float("inf"), float("-inf")],
                        None).where(pd.notnull(df), None)

        # Save as CSV
        os.makedirs("data/downloads", exist_ok=True)
        filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join("data/downloads", filename)
        df.to_csv(filepath, index=False)

        return JSONResponse(content={
            "message": "✅ Cleaning complete",
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records"),
            "download_url": f"/download/{filename}"
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
