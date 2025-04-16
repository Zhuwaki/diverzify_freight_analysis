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

from utils.project_analysis_utils import (
    data_cleaning,
    enrich_invoice_flags,
    uom_cleaning,
    flag_fully_converted_invoices,
    add_freight_per_invoice,
    filter_valid_invoices
)

router = APIRouter()


def prepare_output_data(df):
    # --- Clean numeric fields that may contain errors or text ---
    df['est_estimated_area_cost'] = pd.to_numeric(
        df['est_estimated_area_cost'].astype(
            str).str.extract(r'([-]?[0-9]*\.?[0-9]+)')[0],
        errors='coerce'
    )

    df['est_estimated_cwt_cost'] = pd.to_numeric(
        df['est_estimated_cwt_cost'].astype(
            str).str.extract(r'([-]?[0-9]*\.?[0-9]+)')[0],
        errors='coerce'
    )

    # --- Group by invoice_id and aggregate ---
    df_output_freight = df.groupby(['site', 'invoice_id']).agg(
        # total_quantity=('quantity', 'sum'),
        total_estimated_area_cost=('est_estimated_area_cost', 'sum'),
        total_estimated_cwt_cost=('est_estimated_cwt_cost', 'sum'),
        total_est_lbs=('est_lbs', 'sum'),
        total_est_sqyd=('est_sqyd', 'sum'),
        unique_commodity_group_output=(
            'est_commodity_group', lambda x: x.dropna().unique().tolist()),
        unique_commodity_description_output=(
            'new_commodity_description', lambda x: x.dropna().unique().tolist())
    ).reset_index()

    # View results
    return df_output_freight


def prepare_input_data(df):
    # Get the unique commodity descriptions for each invoice_id amd the freight price from the modelled input
    model_input_freight = df.groupby(['site', 'invoice_id']).agg(
        freight_price=('freight_per_invoice', 'first'),
        unique_commodity_group_input=(
            'new_commodity_group', lambda x: x.dropna().unique().tolist()),
        unique_commodity_description_input=(
            'new_commodity_description', lambda x: x.dropna().unique().tolist())
    ).reset_index()
    return model_input_freight


def cost_uom_format(df):
    df['total_cost'] = df.apply(
        lambda row: row['total_estimated_cwt_cost'] if '1VNL' in row['unique_commodity_group_input'] else (
            row['total_estimated_area_cost'] if '1CBL' in row['unique_commodity_group_input'] else 0
        ),
        axis=1
    )
    # Step 3: Apply conditional logic
    df['total_quantity'] = df.apply(
        lambda row: row['total_est_lbs'] if '1VNL' in row['unique_commodity_group_input'] else (
            row['total_est_sqyd'] if '1CBL' in row['unique_commodity_group_input'] else 0
        ),
        axis=1
    )
    df['UOM'] = df['unique_commodity_group_input'].apply(
        lambda x: 'LBS' if '1VNL' in x else ('SQYD' if '1CBL' in x else None)
    )
    df['freight_ratio'] = (df['total_cost'] / df['freight_price']).round(2)
    return df


def analyze_freight_outliers(df: pd.DataFrame, ratio_col: str = "freight_ratio", plot: bool = True) -> pd.DataFrame:

    # Step 2: Compute IQR bounds
    q1 = df[ratio_col].quantile(0.25)
    q3 = df[ratio_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(
        f"Dynamic Outlier Thresholds:\nLower: {lower_bound:.2f} | Upper: {upper_bound:.2f}")

    # Step 3: Tag outliers
    df["outlier_flag"] = df[ratio_col].apply(
        lambda x: "Lower" if x < lower_bound else (
            "Upper" if x > upper_bound else "Normal")
    )
    df["savings"] = df[ratio_col].apply(lambda x: "Good" if x > 1 else "Bad")
    df["action"] = df[ratio_col].apply(
        lambda x: "Audit" if x > 2 or x < 0.5 else "Analyse")
    return df


def classify_shipment(row):
    uom = row['UOM']
    qty = row['total_quantity']

    if uom == 'LBS':
        return 'FTL' if qty >= 15000 else 'LTL'
    elif uom == 'SQYD':
        rolls = qty / 100
        return 'FTL' if rolls >= 45 else 'LTL'
    else:
        return 'Unknown'  # fallback for unexpected units


def classify_load(df):
    df['shipment_type'] = df.apply(classify_shipment, axis=1)
    return df


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

        # 🔹 Load both dataframes
        estimated_df = read_upload(file1)
        actual_df = read_upload(file2)

        # 📊 Notebook logic — summarizing actual freight
        actual_summary = prepare_output_data(estimated_df)

        # 📈 Summarize estimated freight per invoice
        estimated_summary = prepare_input_data(actual_df)

        # 🧩 Merge both
        merged = estimated_summary.merge(
            actual_summary, on=["site", "invoice_id"], how="outer")
        merged = cost_uom_format(merged)
        merged = classify_load(merged)

        # 📥 Save to file
        output_path = os.path.join(
            "data/downloads", "freight_comparison_report.csv")
        os.makedirs("data/downloads", exist_ok=True)
        merged.to_csv(output_path, index=False)

        return FileResponse(output_path, media_type="text/csv", filename="freight_comparison_report.csv")

    except Exception as e:
        return {"error": str(e)}
