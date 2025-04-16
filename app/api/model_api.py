from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import logging
from datetime import datetime
from utils.model_analysis_utils import estimate_dual_freight_cost, conversion_lookup

router = APIRouter()


@router.post("/batch")
async def estimate_dual_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file format"}

        df.columns = [col.strip().lower().replace(" ", "_")
                      for col in df.columns]

        required = ["site", "invoiced_line_qty", "conversion_code", "po_no"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return {"error": f"Missing columns: {', '.join(missing)}"}

        invalid = df[~df["conversion_code"].str.upper().isin(
            conversion_lookup.keys())]
        if not invalid.empty:
            return {"error": f"Invalid conversion codes: {invalid['conversion_code'].unique().tolist()}"}

        def safe_dual(row):
            try:
                return pd.Series(estimate_dual_freight_cost(
                    quantity=row["invoiced_line_qty"],
                    conversion_code=row["conversion_code"],
                    site=row["site"]
                ))
            except Exception as e:
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
                    "uom": "",
                    'est_pricing_basis': "",
                    'est_cwt_min_applied': "",
                    'est_area_min_applied': "",
                    'est_min_rule_applied': "",
                    'sqyd': "",
                })

        results = df.apply(safe_dual, axis=1)
        results.columns = [f"est_{col}" for col in results.columns]
        final_df = pd.concat([df, results], axis=1)

        os.makedirs("data/downloads", exist_ok=True)
        filename = f"freight_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join("data/downloads", filename)
        final_df.to_csv(filepath, index=False)

        return JSONResponse(content={
            "message": "✅ Model estimation complete",
            "rows": len(final_df),
            "preview": final_df.head(5).fillna("").to_dict(orient="records"),
            "download_url": f"/download/{filename}"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
