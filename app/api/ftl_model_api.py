from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import logging
from datetime import datetime

# Replace with your actual model import
from utils.ftl_modelling_utils import simulate_freight_cost_models_revised
from utils.hybrid_modelling_utils import apply_hybrid_freight_model
from utils.realistic_optimal_utils import apply_realistic_optimal_model
from utils.backup_utils import flag_outliers, compute_line_level_rate_ratio, compute_site_level_freight_ratio, append_group_stats_to_df, export_site_commodity_range_summary


router = APIRouter()


@router.post("/ftlmodel")
async def model_full_truck_load(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})

        # Clean column names
        df.columns = [col.strip().lower().replace(" ", "_")
                      for col in df.columns]

        required_cols = ['invoice_commodity_quantity', 'new_commodity_group', 'applied_rate', 'unit',
                         'raw_invoice_cost', 'invoice_freight_commodity_cost', 'minimum_applied', 'site']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return JSONResponse(status_code=400, content={"error": f"Missing columns: {', '.join(missing_cols)}"})

        # Run the FTL model logic
        df = simulate_freight_cost_models_revised(df)
        df = apply_hybrid_freight_model(df)
        df = apply_realistic_optimal_model(df)
        df = compute_line_level_rate_ratio(df)
        df = compute_site_level_freight_ratio(df)
        df = flag_outliers(df, ['rate_ratio_normal', 'pct_difference'])
     #   df = append_group_stats_to_df(df, ['xgs_rate', 'historical_rate'])

       # df = evaluate_vendor_vs_physical_efficiency(df)

        # Save output
        os.makedirs("data/downloads/ftl", exist_ok=True)
        filename = f"ftl_model_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join("data/downloads/ftl", filename)
        # value_cols = ['xgs_rate', 'historical_rate']
        # output_file = 'site_commodity_rate_summary.csv'

        # summary_table = export_site_commodity_range_summary(
        #     df, value_cols, output_file)

        df.to_csv(filepath, index=False)

        return JSONResponse(content={
            "message": "✅ FTL modeling complete",
            "rows": len(df),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
            "download_url": f"/download/{filename}"
        })

    except Exception as e:
        logging.exception("❌ Exception occurred in /ftlmodel")
        return JSONResponse(status_code=500, content={"error": str(e)})
