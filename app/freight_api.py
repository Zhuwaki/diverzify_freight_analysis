from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import io
import os
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict

# Set up logging
logging.basicConfig(
    filename="freight_api.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(title="Freight Cost Estimator API", version="1.0")

# === Constants ===
class_breakpoints = [
    ("L5C", 0, 499), ("5C", 500, 999), ("1M", 1000, 1999),
    ("2M", 2000, 2999), ("3M", 3000, 4999), ("5M", 5000, 9999),
    ("10M", 10000, 19999), ("20M", 20000, 29999),
    ("30M", 30000, 39999), ("40M", 40000, float("inf")),
]

RATES_CSV_PATH = "freight_rates_updated.csv"
CONVERSION_CSV_PATH = "conversion_table_standardized.csv"

# === Discount structure (can be externalized later) ===
discounts = {
    "CWT": {"SPT": 1, "SPW": 1, "SPJ": 1},
    "SQFT": {"SPT": 1, "SPW": 1, "SPJ": 1},
    "SQYD": {"SPT": 1, "SPW": 1, "SPJ": 1}
}

# === Loaders ===


def load_rate_table_from_csv(filepath: str) -> Dict:
    df = pd.read_csv(filepath)
    df.columns = [col.strip().lower() for col in df.columns]

    # Define valid freight class columns from breakpoints
    valid_class_cols = [c[0].lower() for c in class_breakpoints]

    # Debug available columns
    logging.info(f"📊 Columns in freight rate table: {df.columns.tolist()}")

    required_cols = ["siteid", "unit", "commodity_group"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in rate table.")

    rate_table = {}

    for _, row in df.iterrows():
        site = row["siteid"].strip().upper()
        unit = row["unit"].strip().upper()
        commodity = str(row["commodity_group"]).strip().upper()

        rate_table.setdefault(site, {}).setdefault(
            unit, {}).setdefault(commodity, {})

        for col in df.columns:
            if col not in valid_class_cols:
                continue  # Skip anything that's not a freight class

            rate = row[col]
            try:
                if pd.notna(rate):
                    rate_table[site][unit][commodity][col.upper()
                                                      ] = float(rate)
            except ValueError:
                logging.warning(
                    f"⚠️ Skipped non-numeric rate '{rate}' in column '{col}' for {site}/{unit}/{commodity}")

    logging.info("✅ Rate table loaded successfully.")
    return rate_table


def load_conversion_table(filepath: str) -> Dict:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df["conversion_code"] = df["conversion_code"].str.strip().str.upper()
    logging.info("✅ Conversion table loaded successfully.")
    return {
        row["conversion_code"]: {
            "commodity_group": row["commodity_group"].strip().upper(),
            "uom": row["uom"].strip().upper(),
            "lbs_per_uom": row["lbs_per_uom"]
        }
        for _, row in df.iterrows()
    }


rates = load_rate_table_from_csv(RATES_CSV_PATH)
conversion_lookup = load_conversion_table(CONVERSION_CSV_PATH)

# === Utility Functions ===


def get_priority_class(quantity: float) -> str:
    for class_name, min_q, max_q in class_breakpoints:
        if min_q <= quantity <= max_q:
            return class_name
    raise ValueError("Quantity is out of range.")


def sqft_to_sqyd(sqft: float) -> float:
    return sqft / 9


def normalize_uom(uom: str) -> str:
    uom = uom.strip().upper()
    return "SQFT" if uom == "SF" else "SQYD" if uom == "SY" else uom


def convert_area_to_weight(quantity: float, conversion_code: str):
    code = conversion_code.strip().upper()
    if code not in conversion_lookup:
        raise ValueError(f"Conversion code '{conversion_code}' not found.")
    entry = conversion_lookup[code]
    normalized_uom = normalize_uom(entry["uom"])
    lbs = quantity * entry["lbs_per_uom"]
    return lbs, normalized_uom, entry["commodity_group"]


def get_freight_rate(site: str, unit: str, commodity_group: str, freight_class: str) -> Tuple[Optional[float], Optional[str]]:
    site, unit, commodity_group, freight_class = site.upper(
    ), unit.upper(), commodity_group.upper(), freight_class.upper()
    try:
        if site not in rates:
            return None, f"Site '{site}' not found in rates"
        if unit not in rates[site]:
            return None, f"Unit '{unit}' not available at site '{site}'"
        if commodity_group not in rates[site][unit]:
            return None, f"Commodity group '{commodity_group}' not found under {site}/{unit}"

        rate = rates[site][unit][commodity_group].get(freight_class)
        if rate is None:
            available = list(rates[site][unit][commodity_group].keys())
            return None, f"Class '{freight_class}' not in {site}/{unit}/{commodity_group}. Available: {available}"
        return rate, None
    except Exception as e:
        return None, f"Rate lookup error: {str(e)}"


def estimate_area_based_cost(quantity: float, site: str, commodity_group: str, uom: str):
    uom = normalize_uom(uom)
    if uom == "SQFT":
        quantity = sqft_to_sqyd(quantity)
        uom = "SQYD"
        # NEW RULE: Never calculate area for 1VNL
    if commodity_group.upper() == "1VNL":
        return "Not applicable", None, None, None

    if site not in rates or uom not in rates[site] or commodity_group not in rates[site][uom]:
        logging.info(
            f"ℹ️ Area pricing not available for {site} / {uom} / {commodity_group}")
        return "Not applicable", None, None, None

    freight_class = get_priority_class(quantity)
    rate = rates[site][uom][commodity_group].get(freight_class)
    if rate is None:
        return "Missing class column", freight_class, None, None

    discount = discounts.get(uom, {}).get(site, 1)
    cost = round(rate * discount * quantity, 2)
    return cost, freight_class, rate, discount


def estimate_dual_freight_cost(quantity: float, conversion_code: str, site: str) -> Dict:
    site = site.upper()
    try:
        lbs, original_uom, commodity_group = convert_area_to_weight(
            quantity, conversion_code)
    except Exception as e:
        return {"error": f"Conversion failed: {str(e)}"}

    cwt_quantity = lbs / 100
    freight_class = get_priority_class(lbs)
    cwt_rate, cwt_error = get_freight_rate(
        site, "CWT", commodity_group, freight_class)

    if cwt_error:
        cwt_cost = cwt_error
        cwt_discount = None
    else:
        cwt_discount = discounts.get("CWT", {}).get(site, 1)
        cwt_cost = round(cwt_rate * cwt_discount * cwt_quantity, 2)

    est_sqyd = sqft_to_sqyd(quantity) if original_uom == "SQFT" else quantity
    area_cost, area_freight_class, area_rate, area_discount = estimate_area_based_cost(
        quantity, site, commodity_group, original_uom)

    # Determine pricing basis based on which estimate succeeded
    if isinstance(cwt_cost, (int, float)) and isinstance(area_cost, str):
        pricing_basis = "CWT"
    elif isinstance(area_cost, (int, float)) and isinstance(cwt_cost, str):
        pricing_basis = "AREA"
    elif isinstance(cwt_cost, (int, float)) and isinstance(area_cost, (int, float)):
        pricing_basis = "CWT + AREA"
    else:
        pricing_basis = "Not Applicable"

    return {
        "commodity_group": commodity_group,
        "freight_class_lbs": freight_class,
        "lbs": round(lbs, 2),
        "cwt_quantity": round(cwt_quantity, 2),
        "weight_uom": "lbs",
        "rate_cwt": cwt_rate or "Missing rate",
        "discount_cwt": cwt_discount or "N/A",
        "estimated_cwt_cost": cwt_cost,

        "original_quantity": quantity,
        "original_uom": original_uom,
        "converted_sqyd": round(est_sqyd, 2),
        "freight_class_area": area_freight_class,
        "rate_area": "Not applicable" if area_cost == "Not applicable" else area_rate or "Missing rate",
        "discount_area": area_discount or "N/A",
        "estimated_area_cost": area_cost,
        "area_uom_used": "SQYD" if original_uom in ["SQFT", "SQYD"] else "N/A",
        "est_pricing_basis": pricing_basis
    }


class EstimateRequest(BaseModel):
    quantity: float = Field(..., gt=0)
    conversion_code: str
    site: str


@app.post("/estimate")
def estimate_dual(request: EstimateRequest):
    return estimate_dual_freight_cost(
        quantity=request.quantity,
        conversion_code=request.conversion_code,
        site=request.site
    )


@app.get("/openapi.json")
def get_openapi():
    return app.openapi()


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "version": app.version}


@app.post("/batch")
async def estimate_dual_batch(file: UploadFile = File(...)):
    try:
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

        required = ["siteid", "quantity", "conversion_code", "po_no"]
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
                    quantity=row["quantity"],
                    conversion_code=row["conversion_code"],
                    site=row["siteid"]
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
                    'est_pricing_basis': ""
                })

        results = df.apply(safe_dual, axis=1)
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
            'siteid', 'site', 'supplierid', 'suppliername', 'partnumber',
            'partdescription', 'est_commodity_group', 'new_commodity_description', 'quantity', 'invoice_id', 'invoice_no', 'uom',
            'est_pricing_basis',
            'conversion_code', 'match_supplier', 'est_estimated_area_cost',
            'est_estimated_cwt_cost', 'est_freight_class_area', 'est_freight_class_lbs',
            'est_lbs', 'est_rate_area', 'est_rate_cwt', 'est_sqyd', 'est_uom', 'multiple_parts'
        ]

        export_columns = [
            col for col in model_columns if col in final_df.columns]
        final_df[export_columns].to_csv(final_path, index=False)

        return {
            "filename": file.filename,
            "rows_processed": len(final_df),
            "preview": final_df.head(5).fillna("").to_dict(orient="records"),
            "download_url": f"/download/{output_file}"
        }

    except Exception as e:
        logging.error(f"/batch error: {str(e)}")
        return {"error": str(e)}


@app.get("/download/{filename}")
def download_result(filename: str):
    filepath = os.path.join("downloads", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='text/csv', filename=filename)
    return {"error": "File not found."}
