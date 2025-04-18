import logging
import os
import io
import requests
import pandas as pd
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import math
import time
import os
import json
import requests
from datetime import datetime

from utils.archive.reporting_analysis_utils_adjusted import (prepare_input_data, prepare_output_data,
                                                             classify_load, cost_uom_format,
                                                             analyze_freight_outliers)
router = APIRouter()


@router.post("/report")
async def generate_freight_report(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        def read_upload(upload: UploadFile) -> pd.DataFrame:
            contents = upload.file.read()
            upload.file.seek(0)
            if upload.filename.endswith(".csv"):
                try:
                    return pd.read_csv(io.BytesIO(contents), encoding="utf-8")
                except UnicodeDecodeError:
                    return pd.read_csv(io.BytesIO(contents), encoding="latin1")
            elif upload.filename.endswith((".xls", ".xlsx")):
                return pd.read_excel(io.BytesIO(contents))
            else:
                raise ValueError("Unsupported file format")

        estimated_df = read_upload(file1)
        actual_df = read_upload(file2)

        actual_summary = prepare_output_data(estimated_df)

        estimated_summary = prepare_input_data(actual_df)
        merged = estimated_summary.merge(
            actual_summary, on=["site", "invoice_id"], how="outer")
        merged = cost_uom_format(merged)
        merged = classify_load(merged)
        merged = analyze_freight_outliers(merged)

        os.makedirs("data/downloads", exist_ok=True)
        filename = f"freight_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join("data/downloads", filename)
        merged.to_csv(filepath, index=False)

        return JSONResponse(content={
            "message": "✅ Freight comparison complete",
            "rows": len(merged),
            "preview": merged.head(5).to_dict(orient="records"),
            "download_url": f"/download/{filename}"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
