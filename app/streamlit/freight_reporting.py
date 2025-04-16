import plotly.graph_objects as go
import numpy as np
import plotly.express as px


import streamlit as st
import pandas as pd
from io import BytesIO

# --- App Config ---
st.set_page_config(page_title="Freight Reporting Dashboard", layout="wide")
st.title("🚛 Freight Reporting Dashboard")

st.markdown("""
This dashboard compares freight costs across:
- 📐 **Estimated Freight (Area vs CWT)** — from enriched estimation file
- 📦 **Actual Project Freight** — from raw invoice lines with account 5504

Upload each dataset below:
""")

# --- Uploaders ---
# --- Side-by-side Uploaders ---
col1, col2 = st.columns(2)

with col1:
    est_file = st.file_uploader(
        "📤 Upload *Estimated Freight* CSV", type=["csv"], key="est")

with col2:
    actual_file = st.file_uploader(
        "📥 Upload *Actual Freight (account=5504)* CSV", type=["csv"], key="actual")


# --- Estimated Freight Summary ---
def summarize_estimated_freight(df):
    required = [
        'est_commodity_group', 'new_commodity_description', 'siteid', 'site',
        'est_estimated_area_cost', 'est_estimated_cwt_cost'
    ]
    if not all(col in df.columns for col in required):
        return None

    df['est_estimated_area_cost'] = pd.to_numeric(
        df['est_estimated_area_cost'], errors='coerce')
    df['est_estimated_cwt_cost'] = pd.to_numeric(
        df['est_estimated_cwt_cost'], errors='coerce')

    def summarize(group_cols):
        return df.groupby(group_cols)[
            ['est_estimated_area_cost', 'est_estimated_cwt_cost']
        ].sum().reset_index().rename(columns={
            'est_estimated_area_cost': 'total_area_cost',
            'est_estimated_cwt_cost': 'total_cwt_cost'
        })

    return {
        "group": summarize(['est_commodity_group']),
        "description": summarize(['est_commodity_group', 'new_commodity_description']),
        "site": summarize(['siteid', 'site'])
    }


# --- Actual Project Freight Summary ---
def summarize_project_freight(df):
    required = ['invoice_id', 'freight_per_invoice', 'new_commodity_group',
                'new_commodity_description', 'siteid', 'site']
    if not all(col in df.columns for col in required):
        return None

    df = df.drop_duplicates(subset='invoice_id').copy()
    df['freight_per_invoice'] = pd.to_numeric(
        df['freight_per_invoice'], errors='coerce')

    def summarize(group_cols):
        return df.groupby(group_cols)['freight_per_invoice'].sum().reset_index().rename(
            columns={'freight_per_invoice': 'total_freight_cost'}
        )

    return {
        "group": summarize(['new_commodity_group']),
        "description": summarize(['new_commodity_group', 'new_commodity_description']),
        "site": summarize(['siteid', 'site'])
    }


# --- Style Function for Highlighting Savings ---
def style_savings(df):
    def highlight(val):
        if pd.isna(val):
            return "background-color: lightgray"
        try:
            return "background-color: lightgreen" if val > 0 else "background-color: #ffcccc"
        except:
            return "background-color: lightgray"

    # Ensure savings percent columns are numeric before formatting
    percent_cols = ['cwt_savings_percent', 'area_savings_percent']
    for col in percent_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply background color highlighting
    styled = df.style.applymap(highlight, subset=[
        'cwt_savings', 'area_savings',
        'cwt_savings_percent', 'area_savings_percent'
    ])

    # Format percentage columns and show 'N/A' if missing
    styled = styled.format({
        'cwt_savings_percent': lambda x: f"{int(x)}%" if pd.notna(x) else "N/A",
        'area_savings_percent': lambda x: f"{int(x)}%" if pd.notna(x) else "N/A"
    })

    return styled


# --- Merge Summaries ---
def merge_actual_and_estimated_summaries(df_actual, df_est):
    required_actual = ['invoice_id', 'freight_per_invoice', 'new_commodity_group',
                       'new_commodity_description', 'siteid', 'site']
    if not all(col in df_actual.columns for col in required_actual):
        return None

    df_actual = df_actual.drop_duplicates(subset='invoice_id').copy()
    df_actual['freight_per_invoice'] = pd.to_numeric(
        df_actual['freight_per_invoice'], errors='coerce')

    actual_summary = df_actual.groupby(['siteid', 'new_commodity_group', 'new_commodity_description']).agg(
        unique_freight_total=('freight_per_invoice', 'sum'),
        unique_invoices=('invoice_id', 'nunique')
    ).reset_index()

    required_est = ['siteid', 'est_commodity_group', 'new_commodity_description',
                    'est_estimated_area_cost', 'est_estimated_cwt_cost']
    if not all(col in df_est.columns for col in required_est):
        return None

    df_est['est_estimated_area_cost'] = pd.to_numeric(
        df_est['est_estimated_area_cost'], errors='coerce')
    df_est['est_estimated_cwt_cost'] = pd.to_numeric(
        df_est['est_estimated_cwt_cost'], errors='coerce')

    est_summary = df_est.groupby(['siteid', 'est_commodity_group', 'new_commodity_description'])[
        ['est_estimated_area_cost', 'est_estimated_cwt_cost']].sum().reset_index().rename(columns={
            'est_commodity_group': 'new_commodity_group',
            'est_estimated_area_cost': 'total_area_cost',
            'est_estimated_cwt_cost': 'total_cwt_cost'
        })

    merged = pd.merge(
        actual_summary,
        est_summary,
        on=['siteid', 'new_commodity_group', 'new_commodity_description'],
        how='outer'
    )

    merged['unique_freight_total'] = merged['unique_freight_total'].fillna(0)
    merged['unique_invoices'] = merged['unique_invoices'].fillna(0).astype(int)
    merged['total_area_cost'] = merged['total_area_cost'].fillna(0)
    merged['total_cwt_cost'] = merged['total_cwt_cost'].fillna(0)

    merged['is_cwt_cheaper'] = merged['total_cwt_cost'] < merged['unique_freight_total']
    merged['cwt_savings'] = merged.apply(
        lambda row: row['unique_freight_total'] - row['total_cwt_cost']
        if row['total_cwt_cost'] > 0 else pd.NA,
        axis=1
    )
    merged['cwt_savings_percent'] = merged.apply(
        lambda row: round(
            (row['cwt_savings'] / row['unique_freight_total']) * 100, 2)
        if pd.notna(row['cwt_savings']) and row['unique_freight_total'] > 0 else pd.NA,
        axis=1
    )

    merged['is_area_cheaper'] = merged['total_area_cost'] < merged['unique_freight_total']
    merged['area_savings'] = merged.apply(
        lambda row: row['unique_freight_total'] - row['total_area_cost']
        if row['total_area_cost'] > 0 else pd.NA,
        axis=1
    )
    merged['area_savings_percent'] = merged.apply(
        lambda row: round(
            (row['area_savings'] / row['unique_freight_total']) * 100, 2)
        if pd.notna(row['area_savings']) and row['unique_freight_total'] > 0 else pd.NA,
        axis=1
    )

    def determine_preferred_method(row):
        costs = {
            'CWT': row['total_cwt_cost'],
            'AREA': row['total_area_cost'],
            'ACTUAL': row['unique_freight_total']
        }
        valid_costs = {k: v for k, v in costs.items() if pd.notna(v) and v > 0}
        if not valid_costs:
            return "UNKNOWN"
        min_cost = min(valid_costs.values())
        for method in ['CWT', 'AREA', 'ACTUAL']:
            if method in valid_costs and valid_costs[method] == min_cost:
                return method

    merged['preferred_method'] = merged.apply(
        determine_preferred_method, axis=1)

    round_cols = [
        'unique_freight_total', 'total_area_cost', 'total_cwt_cost',
        'cwt_savings', 'cwt_savings_percent', 'area_savings', 'area_savings_percent'
    ]
    for col in round_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(
                merged[col], errors='coerce').round(0).astype('Int64')

    return merged


# --- Download Helper ---
def csv_download_button(df, label, filename):
    if df is None:
        st.warning(f"⚠️ Cannot download — no data available for: {label}")
        return

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label=label,
        data=buffer,
        file_name=filename,
        mime="text/csv"
    )


# --- Display Logic ---
tabs = []
tab_map = {}
merged_df = None

if est_file:
    try:
        df_est = pd.read_csv(est_file)
        summary = summarize_estimated_freight(df_est)
        if summary:
            tabs.append("📐 Estimated Freight (Area vs CWT)")
            tab_map["📐 Estimated Freight (Area vs CWT)"] = summary
        else:
            st.warning(
                "Estimated file loaded, but required columns are missing.")
    except Exception as e:
        st.error(f"❌ Error reading Estimated file:\n{e}")

if actual_file:
    try:
        df_actual = pd.read_csv(actual_file)
        summary = summarize_project_freight(df_actual)
        if summary:
            tabs.append("📦 Actual Project Freight")
            tab_map["📦 Actual Project Freight"] = summary
        else:
            st.warning("Actual file loaded, but required columns are missing.")
    except Exception as e:
        st.error(f"❌ Error reading Actual file:\n{e}")

if est_file and actual_file and 'df_est' in locals() and 'df_actual' in locals():
    try:
        merged_df = merge_actual_and_estimated_summaries(df_actual, df_est)
        if merged_df is not None:
            tabs.append("🔀 Combined Freight View")
    except Exception as e:
        st.error(f"❌ Error generating merged view:\n{e}")

if tabs:
    main_tab = st.tabs(tabs)
    keys = ["group", "description", "site"]

    if merged_df is not None:
        with main_tab[tabs.index("🔀 Combined Freight View")]:
            st.markdown("### 🔀 Combined Freight View")
            styled_df = style_savings(merged_df)
            st.write(styled_df)
            # --- Optional Graphs (Safe block) ---
            try:
                st.markdown(
                    "### 📊 Freight Cost Comparison Views (Interactive)")

                groupings = {
                    "Commodity type": "new_commodity_group",
                    "Commodity description": "new_commodity_description",
                    "Site location": "siteid"
                }

                cols = st.columns(3)

                for (title, group_col), col in zip(groupings.items(), cols):
                    with col:
                        # Aggregate and reshape
                        df_plot = (
                            merged_df
                            .groupby(group_col)[
                                ["unique_freight_total",
                                    "total_area_cost", "total_cwt_cost"]
                            ]
                            .sum()
                            .reset_index()
                        )

                        if df_plot.empty or df_plot[["unique_freight_total", "total_area_cost", "total_cwt_cost"]].sum().sum() == 0:
                            col.warning(f"No data to display for: {title}")
                            continue

                        # Convert to long-form (melted) for Plotly Express
                        df_melted = df_plot.melt(
                            id_vars=[group_col],
                            value_vars=["unique_freight_total",
                                        "total_area_cost", "total_cwt_cost"],
                            var_name="Cost Type",
                            value_name="Cost (USD '000)"
                        )
                        df_melted["Cost (USD '000)"] = (
                            df_melted["Cost (USD '000)"] / 1000)
                        df_melted["Cost Type"] = df_melted["Cost Type"].map({
                            "unique_freight_total": "Actual Freight",
                            "total_area_cost": "Estimated Area",
                            "total_cwt_cost": "Estimated CWT"
                        })

                        fig = px.bar(
                            df_melted,
                            x=group_col,
                            y="Cost (USD '000)",
                            color="Cost Type",
                            barmode="group",
                            # text_auto=True,
                            labels={group_col: title},
                            height=350
                        )
                        fig.update_layout(
                            title=title,
                            xaxis_tickangle=45,
                            margin=dict(l=20, r=20, t=30, b=40),
                            showlegend=(group_col == "siteid")
                        )
                        col.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"⚠️ Could not generate interactive graphs: {e}")

    csv_download_button(
        merged_df,
        "⬇️ Download Combined Freight View",
        "combined_freight_summary.csv"
    )

    for tab_name, summary_dict in tab_map.items():
        idx = tabs.index(tab_name)
        with main_tab[idx]:
            st.markdown(f"### {tab_name}")
            sub_tabs = st.tabs(["🧱 By Group", "📦 By Description", "📍 By Site"])
            for i, key in enumerate(keys):
                with sub_tabs[i]:
                    df_summary = summary_dict[key]
                    st.dataframe(df_summary)
                    csv_download_button(

                        df_summary,
                        f"⬇️ Download {key.capitalize()} CSV",
                        f"{tab_name.lower().replace(' ', '_')}_{key}.csv"
                    )
else:
    st.info("📈 Upload one or both files above to get started.")
