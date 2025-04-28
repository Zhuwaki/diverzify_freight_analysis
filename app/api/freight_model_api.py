from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import logging
from datetime import datetime


from utils.invoice_freight_utils import (
    prepare_invoice_freight_summary,
    standardize_input_data
)

router = APIRouter()


@router.post("/freightmodel")
async def estimate_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file format"}

        # Clean column names
        df.columns = [col.strip().lower().replace(" ", "_")
                      for col in df.columns]

        required_cols = ["site", "invoiced_line_qty",
                         "conversion_code", "po_no", "inv_uom", "new_commodity_group"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing columns: {', '.join(missing_cols)}"}

        # Step 1: Estimate line-level freight costs
        # freight_estimates = df.apply(estimate_line_freight_cost_final, axis=1)
        # df = pd.concat([df, freight_estimates], axis=1)

        # Step 2: Aggregate invoice-level freight costs
       # df = estimate_total_freight(df)
        # df = standardize_input_data(df)
        df = prepare_invoice_freight_summary(df)

        # Save output
        os.makedirs("data/downloads", exist_ok=True)
        filename = f"freight_model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join("data/downloads", filename)
        df.to_csv(filepath, index=False)

        return JSONResponse(content={
            "message": "✅ Freight estimation complete",
            "rows": len(df),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
            "download_url": f"/download/{filename}"
        })

    except Exception as e:
        logging.exception("❌ Exception occurred in /freightmodel")
        return JSONResponse(status_code=500, content={"error": str(e)})
