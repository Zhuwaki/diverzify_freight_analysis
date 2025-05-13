# --- FastAPI Backend ---
# File: api.py

import os
import base64
import io
import logging
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for plotting

router = APIRouter()

# Load freight rate table once
RATE_PATH = os.path.join(os.path.dirname(
    __file__), "freight_rates_operating.csv")
rates_df = pd.read_csv(RATE_PATH)
rates_df.columns = rates_df.columns.str.strip()

RATE_TIERS = [
    (5, "L5C"), (100, "5C"), (1000, "1M"), (2000, "2M"), (3000, "3M"),
    (5000, "5M"), (10000, "10M"), (20000, "20M"), (30000, "30M"), (40000, "40M")
]

MAX_X_AXIS = {
    "1VNL": 44000,
    "1CBL": 10000,
    "1CPT": 10000
}

TRUCK_CAPACITY = {
    "1CBL": 4000,
    "1CPT": 3500,
    "1VNL": 40000
}

FTL_VENDOR_THRESHOLD = {
    "1CBL": 2200,
    "1CPT": 2200,
    "1VNL": 20000
}

VALID_UOM = {
    "1VNL": "CWT",
    "1CBL": "SQYD",
    "1CPT": "SQYD"
}


class FreightEstimateInput(BaseModel):
    site: str
    commodity: Literal["1CBL", "1VNL", "1CPT"]
    quantity: float
    uom: Literal["SQYD", "CWT"]


class FreightEstimateOutput(BaseModel):
    ltl_cost: float
    ftl_cost: float
    optimal_mode: str
    ltl_rate: float
    freight_class: str
    plot_base64: str


def build_ltl_curve(site: str, commodity: str, uom: str, qty: float, step: int = 1):
    logging.info(f"Building LTL curve for {site}, {commodity}, {uom}, {qty}")
    filtered = rates_df[
        (rates_df["site"] == site)
        & (rates_df["commodity_group"] == commodity)
        & (rates_df["unit"] == uom)
    ]
    if filtered.empty:
        raise ValueError("No matching rate data found.")

    row = filtered.iloc[0]
    curve = []
    rate_breaks = []
    applied_rate = 0.0
    applied_class = ""
    prev_bound = 0

    for bound, col in RATE_TIERS:
        if col in row and not pd.isna(row[col]):
            rate = row[col] / 100 if commodity == "1VNL" else row[col]
            for q in range(prev_bound + step, bound + 1, step):
                curve.append((q, q * rate))
                if qty <= q and applied_rate == 0.0:
                    applied_rate = rate
                    applied_class = col
            rate_breaks.append(bound)
            prev_bound = bound

    max_x = MAX_X_AXIS.get(commodity)
    if max_x:
        capped = [(q, c) for q, c in curve if q <= max_x]
        if len(capped) >= 2:
            curve = capped
        rate_breaks = [b for b in rate_breaks if b <= max_x]

    ftl_cost = float(row["FTL"]) if "FTL" in row and not pd.isna(
        row["FTL"]) else 0.0
    return curve, applied_rate, applied_class, ftl_cost, rate_breaks


def interpolate_cost(curve, qty):
    if not curve:
        return 0.0
    x, y = zip(*curve)
    if qty < min(x):
        return y[0]
    elif qty > max(x):
        return y[-1]
    return float(np.interp(qty, x, y))


def find_intersection(curve, flat_y):
    for i in range(1, len(curve)):
        q1, c1 = curve[i - 1]
        q2, c2 = curve[i]
        if (c1 - flat_y) * (c2 - flat_y) <= 0:
            if c2 != c1:
                m = (c2 - c1) / (q2 - q1)
                x_int = q1 + (flat_y - c1) / m
            else:
                x_int = q1
            return x_int, flat_y
    return None


# ... existing setup unchanged ...

# Update only generate_plot with offset annotation labels

def generate_plot(curves, qty, ltl_cost, ftl_cost, user_cost, max_x, rate_breaks, commodity, intersection=None, vendor_intersection=None, truck_intersection=None):
    logging.info("Generating plot")
    fig, ax = plt.subplots()
    for mode, curve in curves.items():
        q, c = zip(*curve)
        if mode == "FTL":
            ax.plot(q, c, label=f"{mode} Cost Curve",
                    linestyle="--", color="orange", linewidth=2)
        else:
            ax.plot(q, c, label=f"{mode} Cost Curve")

    for x in rate_breaks:
        ax.axvline(x=x, linestyle=":", color="gray", alpha=0.4)

    truck_cap = TRUCK_CAPACITY.get(commodity)
    vendor_threshold = FTL_VENDOR_THRESHOLD.get(commodity)

    if truck_cap:
        ax.axvline(x=truck_cap, linestyle='--', color='red',
                   alpha=0.6, label='Truck Capacity')
    if vendor_threshold:
        ax.axvline(x=vendor_threshold, linestyle='--', color='blue',
                   alpha=0.6, label='Vendor Threshold')

    if intersection:
        x_int, y_int = intersection
        ax.scatter(x_int, y_int, color='purple', s=60, label='LTL = FTL')
        ax.annotate(f"LTL = FTL\n({int(x_int)}, ${int(y_int)})",
                    xy=(x_int, y_int),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=8, color='purple', ha='left')

    if vendor_intersection:
        vx, vy = vendor_intersection
        ax.scatter(vx, vy, color='blue', s=60)
        ax.annotate(f"Vendor Threshold\n({int(vx)}, ${int(vy)})",
                    xy=(vx, vy),
                    xytext=(10, -20),
                    textcoords="offset points",
                    fontsize=8, color='blue', ha='left')

    if truck_intersection:
        tx, ty = truck_intersection
        ax.scatter(tx, ty, color='green', s=60)
        ax.annotate(f"Truck Capacity\n({int(tx)}, ${int(ty)})",
                    xy=(tx, ty),
                    xytext=(10, -40),
                    textcoords="offset points",
                    fontsize=8, color='green', ha='left')

    ax.scatter(qty, user_cost, color='red', label="User Query", zorder=5)
    ax.set_xlim(0, max_x)
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Freight Cost")
    ax.set_title("Freight Cost Curves")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@router.post("/estimate", response_model=FreightEstimateOutput)
def estimate_freight(input: FreightEstimateInput):
    try:
        expected_uom = VALID_UOM.get(input.commodity)
        if expected_uom != input.uom:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid unit of measure for {input.commodity}. Expected: {expected_uom}"
            )

        ltl_curve, ltl_rate, freight_class, ftl_cost, rate_breaks = build_ltl_curve(
            input.site, input.commodity, input.uom, input.quantity, step=1)

        if not ltl_curve or len(ltl_curve) < 2:
            raise HTTPException(
                status_code=400, detail="Insufficient data to plot cost curve.")

        ltl_cost = interpolate_cost(ltl_curve, input.quantity)
        ftl_line = [(min(q for q, _ in ltl_curve), ftl_cost),
                    (max(q for q, _ in ltl_curve), ftl_cost)]
        optimal_mode = "FTL" if ftl_cost < ltl_cost else "LTL"
        max_x = MAX_X_AXIS.get(input.commodity, max(q for q, _ in ltl_curve))
        intersection = find_intersection(ltl_curve, ftl_cost)

        vendor_threshold = FTL_VENDOR_THRESHOLD.get(input.commodity)
        vendor_intersection = None
        if vendor_threshold:
            vendor_intersection = (
                vendor_threshold, interpolate_cost(ltl_curve, vendor_threshold))

        truck_cap = TRUCK_CAPACITY.get(input.commodity)
        truck_intersection = None
        if truck_cap:
            truck_intersection = (
                truck_cap, interpolate_cost(ltl_curve, truck_cap))

        plot_b64 = generate_plot({"LTL": ltl_curve, "FTL": ftl_line},
                                 input.quantity, ltl_cost, ftl_cost,
                                 ltl_cost if optimal_mode == "LTL" else ftl_cost,
                                 max_x, rate_breaks, input.commodity,
                                 intersection=intersection,
                                 vendor_intersection=vendor_intersection,
                                 truck_intersection=truck_intersection)

        return FreightEstimateOutput(
            ltl_cost=ltl_cost,
            ftl_cost=ftl_cost,
            optimal_mode=optimal_mode,
            freight_class=freight_class,
            ltl_rate=ltl_rate,
            plot_base64=plot_b64,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
