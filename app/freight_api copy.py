from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import io
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename="freight_api.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

class_breakpoints = [
    ("L5C", 0, 499), ("5C", 500, 999), ("1M", 1000, 1999),
    ("2M", 2000, 2999), ("3M", 3000, 4999), ("5M", 5000, 9999),
    ("10M", 10000, 19999), ("20M", 20000, 29999),
    ("30M", 30000, 39999), ("40M", 40000, float("inf")),
]


def load_rate_table_from_csv(filepath: str):
    df = pd.read_csv(filepath)
    df.columns = [col.strip().lower() for col in df.columns]
    rate_table = {}

    for _, row in df.iterrows():
        site = row["siteid"].upper()
        unit = row["unit"].upper()
        commodity = str(row["commodity_group"]).strip().upper()

        rate_table.setdefault(site, {}).setdefault(
            unit, {}).setdefault(commodity, {})

        for col in df.columns:
            if col in ["site", "siteid", "unit", "commodity_group", "unitclass", "freightclass", "oldfreightclass", "commoditydescription"]:
                continue
            rate = row[col]
            if pd.notna(rate):
                rate_table[site][unit][commodity][col.upper()] = float(rate)
    logging.info("✅ Rate table loaded successfully.")
    return rate_table


def load_conversion_table(filepath: str):
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


RATES_CSV_PATH = "freight_rates_updated.csv"
CONVERSION_CSV_PATH = "conversion_table_standardized.csv"
rates = load_rate_table_from_csv(RATES_CSV_PATH)
conversion_lookup = load_conversion_table(CONVERSION_CSV_PATH)

discounts = {
    "CWT": {"SPT": 1, "SPW": 1, "SPJ": 1},
    "SQFT": {"SPT": 1, "SPW": 1, "SPJ": 1},
    "SQYD": {"SPT": 1, "SPW": 1, "SPJ": 1}
}


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


def estimate_area_based_cost(quantity: float, site: str, commodity_group: str, uom: str):
    uom = normalize_uom(uom)
    if uom == "SQFT":
        quantity = sqft_to_sqyd(quantity)
        uom = "SQYD"

    if site not in rates or uom not in rates[site] or commodity_group not in rates[site][uom]:
        logging.warning(
            f"❌ Missing rate entry for {site} / {uom} / {commodity_group}")
        return "Missing rate entry", None, None, None

    freight_class = get_priority_class(quantity)
    rate_group = rates[site][uom][commodity_group]
    rate = rate_group.get(freight_class)

    if rate is None:
        logging.warning(
            f"❌ Missing class {freight_class} in rate group for {site} / {uom} / {commodity_group}")
        return "Missing class column", freight_class, None, None

    discount = discounts.get(uom, {}).get(site, 1)
    cost = round(rate * discount * quantity, 2)
    return cost, freight_class, rate, discount


def estimate_dual_freight_cost(quantity: float, conversion_code: str, site: str):
    site = site.upper()
    try:
        lbs, original_uom, commodity_group = convert_area_to_weight(
            quantity, conversion_code)
    except Exception as e:
        logging.error(f"Conversion error: {str(e)}")
        raise

    cwt_quantity = lbs / 100
    freight_class = get_priority_class(lbs)
    # Area conversion
    est_sqyd = sqft_to_sqyd(quantity) if original_uom == "SQFT" else quantity

    try:
        rate_group = rates[site]["CWT"][commodity_group]
        cwt_rate = rate_group.get(freight_class)
        if cwt_rate is None:
            logging.warning(
                f"❌ Missing CWT class {freight_class} for {commodity_group} at {site}")
            cwt_cost = "Missing class column"
        else:
            cwt_discount = discounts.get("CWT", {}).get(site, 1)
            cwt_cost = round(cwt_rate * cwt_discount * cwt_quantity, 2)
    except KeyError as e:
        logging.warning(f"❌ Missing CWT entry for {site} / {commodity_group}")
        cwt_cost = "Missing rate entry"
        cwt_rate = None
        cwt_discount = None

    area_cost, area_freight_class, area_rate, area_discount = estimate_area_based_cost(
        quantity, site, commodity_group, original_uom)

    return {
        "commodity_group": commodity_group,
        "lbs": round(lbs, 2),
        "freight_class_lbs": freight_class if isinstance(cwt_rate, (int, float)) else "Missing rate",
        "rate_cwt": cwt_rate if isinstance(cwt_rate, (int, float)) else "Missing rate",
        "discount_cwt": cwt_discount if isinstance(cwt_discount, (int, float)) else "Missing rate",
        "estimated_cwt_cost": cwt_cost,

        "sqyd": round(est_sqyd, 2),
        "freight_class_area": area_freight_class if isinstance(area_rate, (int, float)) else "Missing rate",
        "rate_area": area_rate if isinstance(area_rate, (int, float)) else "Missing rate",
        "discount_area": area_discount if isinstance(area_discount, (int, float)) else "Missing rate",
        "estimated_area_cost": area_cost,

        "uom": original_uom,
    }


def make_column_names_unique(columns):
    seen = {}
    result = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result


@app.post("/estimate")
def estimate_dual(data: dict):
    try:
        return estimate_dual_freight_cost(
            quantity=data["quantity"],
            conversion_code=data["conversion_code"],
            site=data["site"]
        )
    except Exception as e:
        logging.error(f"/estimate error: {str(e)}")
        return {"error": str(e)}


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
        df.columns = make_column_names_unique(df.columns)

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
                    "uom": ""
                })

        results = df.apply(safe_dual, axis=1)
        results.columns = [f"est_{col}" for col in results.columns]
        final_df = pd.concat([df, results], axis=1)

        os.makedirs("downloads", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"freight_dual_results_{timestamp}.csv"
        final_path = os.path.join("downloads", output_file)
        final_df.to_csv(final_path, index=False)

        # Define the desired export columns
        model_columns = [
            'project_id', 'project_name', 'po_no', 'account', 'account_description',
            'siteid', 'site', 'supplierid', 'suppliername', 'partnumber',
            'partdescription', 'est_commodity_group', 'new_commodity_description', 'quantity', 'invoice_id', 'invoice_no', 'uom',
            'conversion_code', 'match_supplier', 'est_estimated_area_cost',
            'est_estimated_cwt_cost', 'est_freight_class_area', 'est_freight_class_lbs',
            'est_lbs', 'est_rate_area', 'est_rate_cwt', 'est_sqyd', 'est_uom'
        ]

        # Filter only available columns before saving
        export_columns = [
            col for col in model_columns if col in final_df.columns]
        final_df[export_columns].to_csv(final_path, index=False)

        logging.info(
            f"✅ Batch processed: {output_file} with {len(final_df)} rows")

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
