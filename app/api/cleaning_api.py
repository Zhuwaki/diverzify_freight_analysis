from utils.data_cleaning_utils import (

    map_manufacturer,
    map_commodity,
    create_conversion_code,
    classify_line_uom,
    classify_freight_lines,
    classify_parts_and_commodities,
    classify_priority_products_2008,
    add_freight_per_invoice,
    priority_product_composition,
    add_invoice_total,
    classify_priority_commodities,
    filter_valid_invoices,
    filter_sample_invoices,
    classify_priority_invoice_with_labels,
    map_supplier_characteristics


)
import traceback
import orjson
import os
import io
import pandas as pd
from datetime import datetime
from fastapi import UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
import numpy as np


router = APIRouter()
conversion_csv_path = "data/input/freight_model/conversion_table_standardized.csv"


@router.post("/clean_input_file")
async def prepare_raw_input_file(file: UploadFile = File(...)):
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

        df = map_manufacturer(df)
        df = map_commodity(df)
        df = classify_line_uom(df)
        df = create_conversion_code(df)
        df = classify_freight_lines(df)
        df = classify_parts_and_commodities(df)
        df = classify_priority_commodities(df)
        df = classify_priority_products_2008(df)
        df = add_freight_per_invoice(df)
        df = priority_product_composition(df)
        df = add_invoice_total(df)
        # df = filter_valid_invoices(df)
        df = classify_priority_invoice_with_labels(df)
        df = map_supplier_characteristics(df)
        # df = filter_sample_invoices(df)

        # First: Replace infinities
        df = df.replace([np.inf, -np.inf], np.nan)

        # Second: Replace NaNs with None using .mask
        df = df.mask(df.isna(), None)
        # 🛡️ Fix numpy.bool_ to Python bool
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(bool)

        # Save as CSV
        os.makedirs("data/downloads/cleaning", exist_ok=True)
        filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join("data/downloads/cleaning", filename)
        df.to_csv(filepath, index=False)
        # Fill numeric columns NaN with 0
        numeric_cols = df.select_dtypes(
            include=['float64', 'float32', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Convert DataFrame to JSON
        # 🧪 Scan entire DataFrame before JSONResponse

        # 🛡️ Step 1: Flag bad rows (rows with NaN, inf, or -inf)
        bad_rows = df[
            df.isin([np.inf, -np.inf]).any(axis=1) |
            df.isnull().any(axis=1)
        ]

        if not bad_rows.empty:
            print(
                f"⚠️ Found {len(bad_rows)} bad rows before JSON serialization!")
            bad_rows.to_csv(
                'data/downloads/cleaning/bad_rows_before_json.csv', index=False)
        else:
            print("✅ No bad rows found before JSON serialization.")

        # 🛡️ Step 2: Flag bad columns (columns with NaN, inf, or -inf)
        bad_columns_summary = []

        for col in df.columns:
            # Check for NaN and inf in each column
            col_is_nan = df[col].isnull().sum()
            col_is_inf = df[col].isin([np.inf, -np.inf]).sum()
            total = len(df)

            if col_is_nan > 0 or col_is_inf > 0:
                bad_columns_summary.append({
                    'column_name': col,
                    'nan_count': col_is_nan,
                    'inf_count': col_is_inf,
                    'total_rows': total,
                    'nan_pct': round((col_is_nan / total) * 100, 2),
                    'inf_pct': round((col_is_inf / total) * 100, 2),
                    'dtype': df[col].dtype
                })

        # Output bad columns summary
        if bad_columns_summary:
            bad_columns_df = pd.DataFrame(bad_columns_summary)
            bad_columns_df.to_csv(
                'data/downloads/cleaning/bad_columns_summary.csv', index=False)
            print("⚠️ Bad columns detected and exported.")
        else:
            print("✅ No bad columns detected.")

        # Final protection against JSON crash
        numeric_cols = df.select_dtypes(
            include=['float64', 'float32', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return JSONResponse(content={
            "message": "✅ Cleaning complete",
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records"),
            "download_url": f"/download/{filename}"
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
