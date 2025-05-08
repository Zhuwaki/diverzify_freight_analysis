# --- FastAPI Backend ---
# File: api.py

from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

router = APIRouter()

# Load freight rate table once
RATE_PATH = os.path.join(os.path.dirname(
    __file__), "freight_rates_operating.csv")
rates_df = pd.read_csv(RATE_PATH)
rates_df.columns = rates_df.columns.str.strip()

RATE_TIERS = [
    (5, "L5C"),
    (100, "5C"),
    (1000, "1M"),
    (2000, "2M"),
    (3000, "3M"),
    (5000, "5M"),
    (10000, "10M"),
    (20000, "20M"),
    (30000, "30M"),
    (40000, "40M")
]

MAX_X_AXIS = {
    "1VNL": 44000,  # LBS
    "1CBL": 4400,   # SQYD
    "1CPT": 4400    # SQYD
}

VALID_UOM = {
    "1VNL": "LBS",
    "1CBL": "SQYD",
    "1CPT": "SQYD"
}


class FreightEstimateInput(BaseModel):
    site: str
    commodity: Literal["1CBL", "1VNL", "1CPT"]
    quantity: float
    uom: Literal["SQYD", "LBS"]


class FreightEstimateOutput(BaseModel):
    ltl_cost: float
    ftl_cost: float
    optimal_mode: str
    plot_base64: str
    ltl_rate: float
    freight_class: str


def build_ltl_curve(site: str, commodity: str, uom: str, qty: float, step: int = 100):
    filtered = rates_df[
        (rates_df["site"] == site)
        & (rates_df["commodity_group"] == commodity)
        & (rates_df["unit"] == uom)
    ]
    if filtered.empty:
        raise ValueError("No matching rate data found.")

    row = filtered.iloc[0]
    curve = []
    applied_rate = 0.0
    applied_class = ""
    prev_bound = 0

    for bound, col in RATE_TIERS:
        if col in row and not pd.isna(row[col]):
            rate = row[col]
            for q in range(prev_bound + step, bound + 1, step):
                curve.append((q, q * rate))
                if qty <= q and applied_rate == 0.0:
                    applied_rate = rate
                    applied_class = col
            prev_bound = bound

    # Cap x-axis using commodity max
    max_x = MAX_X_AXIS.get(commodity)
    if max_x:
        capped = [(q, c) for q, c in curve if q <= max_x]
        if len(capped) >= 2:
            curve = capped

    ftl_cost = float(row["FTL"]) if "FTL" in row and not pd.isna(
        row["FTL"]) else 0.0
    return curve, applied_rate, applied_class, ftl_cost


def interpolate_cost(curve, qty):
    if not curve:
        return 0.0
    x, y = zip(*curve)
    if qty < min(x):
        return y[0]
    elif qty > max(x):
        return y[-1]
    return float(np.interp(qty, x, y))


def generate_plot(curves, qty, ltl_cost, ftl_cost, user_cost, max_x):
    fig, ax = plt.subplots()
    for mode, curve in curves.items():
        q, c = zip(*curve)
        if mode == "FTL":
            ax.plot(q, c, label=f"{mode} Cost Curve",
                    linestyle="--", color="orange", linewidth=2)
        else:
            ax.plot(q, c, label=f"{mode} Cost Curve")
    ax.scatter(qty, user_cost, color='red', label="User Query", zorder=5)
    ax.set_xlim(0, max_x)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Freight Cost")
    ax.legend()
    ax.set_title("Freight Cost Curves")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@router.post("/estimate", response_model=FreightEstimateOutput)
def estimate_freight(input: FreightEstimateInput):
    try:
        # Validate UOM compatibility
        expected_uom = VALID_UOM.get(input.commodity)
        if expected_uom != input.uom:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid unit of measure for {input.commodity}. Expected: {expected_uom}"
            )

        # Attempt to build curve
        try:
            ltl_curve, ltl_rate, freight_class, ftl_cost = build_ltl_curve(
                input.site, input.commodity, input.uom, input.quantity, step=100)
        except ValueError as ve:
            print(f"[ERROR] {ve}")
            raise HTTPException(status_code=400, detail=str(ve))

        # Ensure curve is not empty or invalid
        if not ltl_curve or len(ltl_curve) < 2:
            print("[ERROR] LTL curve has insufficient data points.")
            raise HTTPException(
                status_code=400, detail="Insufficient data to plot cost curve.")

        ltl_cost = interpolate_cost(ltl_curve, input.quantity)
        ftl_line = [(min(q for q, _ in ltl_curve), ftl_cost),
                    (max(q for q, _ in ltl_curve), ftl_cost)]
        optimal_mode = "FTL" if ftl_cost < ltl_cost else "LTL"
        max_x = MAX_X_AXIS.get(input.commodity, max(q for q, _ in ltl_curve))
        plot_b64 = generate_plot({"LTL": ltl_curve, "FTL": ftl_line},
                                 input.quantity, ltl_cost, ftl_cost,
                                 ltl_cost if optimal_mode == "LTL" else ftl_cost,
                                 max_x)

        return FreightEstimateOutput(
            ltl_cost=ltl_cost,
            ftl_cost=ftl_cost,
            optimal_mode=optimal_mode,
            plot_base64=plot_b64,
            ltl_rate=ltl_rate,
            freight_class=freight_class
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
