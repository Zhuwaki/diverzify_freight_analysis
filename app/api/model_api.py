from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import io
import os
import logging
from datetime import datetime
from utils.model_analysis_utils import (
    estimate_dual_freight_cost,
    conversion_lookup,

)

# Clear any existing handlers first
import logging

# Remove existing handlers if any
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# ✅ Log to BOTH terminal and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("freight_api.log"),   # ← logs to file
        logging.StreamHandler()                   # ← logs to terminal
    ]
)

router = APIRouter()


class EstimateRequest(BaseModel):
    quantity: float = Field(..., gt=0)
    conversion_code: str
    site: str


@router.post("/batch")
async def estimate_dual_batch(file: UploadFile = File(...)):
    try:
        logging.info("📥 Received file in /batch: %s", file.filename)
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename.endswith((".xls", ".xlsx")):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file type. Please upload .csv or .xlsx"}

        df.columns = [col.strip().lower().replace(" ", "_")
                      for col in df.columns]

        required = ["site", "invoiced_line_qty", "conversion_code", "po_no"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            return {"error": f"Missing columns: {', '.join(missing)}"}

        invalid_codes = df[~df["conversion_code"].str.upper().isin(
            conversion_lookup.keys())]
        if not invalid_codes.empty:
            return {"error": f"Invalid conversion codes: {invalid_codes['conversion_code'].unique().tolist()}"}

        def safe_dual(row):
            try:
                return pd.Series(estimate_dual_freight_cost(
                    quantity=row["invoiced_line_qty"],
                    conversion_code=row["conversion_code"],
                    site=row["site"]
                ))
            except Exception as e:
                logging.error(f"❌ Row error: {str(e)}")
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

        logging.info("🧹 Starting freight estimation pipeline...")
        logging.info("Rows in file: %s", df.shape[0])
        results = df.apply(safe_dual, axis=1)
        logging.info("✅ Freight estimation complete.")

        results.columns = [f"est_{col}" for col in results.columns]
        final_df = pd.concat([df, results], axis=1)

        os.makedirs("downloads", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"freight_dual_results_{timestamp}.csv"
        final_path = os.path.join("downloads", output_file)
        # Define the desired export columns
        model_columns = [
            'est_pricing_basis',
            'project_id', 'project_name', 'po_no', 'account', 'account_description',
            'site', 'site_name', 'supplierid', 'suppliername', 'partnumber',
            'partdescription', 'est_commodity_group', 'new_commodity_description', 'invoiced_line_qty', 'invoice_id', 'invoice_no', 'uom',
            'est_pricing_basis',
            'conversion_code', 'match_supplier', 'est_estimated_area_cost',
            'est_estimated_cwt_cost', 'est_freight_class_area', 'est_freight_class_lbs',
            'est_lbs', 'est_rate_area', 'est_rate_cwt', 'est_uom', 'multiple_parts', 'est_cwt_min_applied',
            'est_area_min_applied', 'est_min_rule_applied', 'est_raw_area_cost',
            'est_raw_cwt_cost', 'est_sqyd',

        ]

        export_columns = [
            col for col in model_columns if col in final_df.columns]

        print("🧪 FINAL COLUMNS:", final_df.columns.tolist())
        if 'est_cwt_min_applied' in final_df.columns and 'est_area_min_applied' in final_df.columns:
            print("🧪 SAMPLE VALUES:")
            print(final_df[['est_cwt_min_applied',
                  'est_area_min_applied', 'est_min_rule_applied']].head())
        else:
            print(
                "🧪 MISSING one or both of: 'est_cwt_min_applied', 'est_area_min_applied'")

        final_df[export_columns].to_csv(final_path, index=False)
        print(final_df[['est_cwt_min_applied', 'est_area_min_applied',
              'est_min_rule_applied']].head())

        return {
            "filename": file.filename,
            "rows_processed": len(final_df),
            "preview": final_df.head(5).fillna("").to_dict(orient="records"),
            "download_url": f"/download/{output_file}"
        }

    except Exception as e:
        logging.error(f"/batch error: {str(e)}")
        return {"error": str(e)}
