from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import logging
from datetime import datetime


from utils.ltl_modelling_utils import (
    estimate_invoice_freight,
    filter_valid_priority_lines,
    rank_freight_class
)

router = APIRouter()


@router.post("/ltlmodel")
async def model_partial_truck_load(file: UploadFile = File(...)):
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

        df = estimate_invoice_freight(df)
        # df = calibrate_surcharge(df)
        # df = compute_market_rates(df)
        # df = compute_freight_and_rate_ratios(df)
       # df = flag_market_cost_outliers(df)
        df = filter_valid_priority_lines(df)
        df = rank_freight_class(df, class_column='freight_class')

        # Save output
        os.makedirs("data/downloads/ltl", exist_ok=True)
        filename = f"ltl_model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join("data/downloads/ltl", filename)
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
