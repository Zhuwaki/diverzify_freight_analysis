import streamlit as st
import pandas as pd
import os

# Load invoice data


@st.cache
def load_data():
    path = "../data/input/Freight_Cost_Analysis_CY2024-03.25.csv"
    if not os.path.exists(path):
        st.error(f"❌ File not found: {path}")
        return pd.DataFrame()  # return empty DataFrame for safety
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


df = load_data()

# Title
st.title("📄 Invoice Lookup Tool")

# Input
invoice_id = st.text_input("Enter Invoice ID", "")

if invoice_id:
    try:
        invoice_id = int(invoice_id)
        if "invoice_id" not in df.columns:
            st.error("⚠️ 'estimate_invoice_id' column not found in data.")
        else:
            results = df[df["invoice_id"] == invoice_id]

            if results.empty:
                st.warning("No items found for this Invoice ID.")
            else:
                st.success(f"Found {len(results)} item(s).")
                st.dataframe(results)
    except ValueError:
        st.error("Please enter a valid numeric Invoice ID.")

# Option to download filtered results
if 'results' in locals() and not results.empty:
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results as CSV",
        data=csv,
        file_name=f"invoice_{invoice_id}_items.csv",
        mime="text/csv"
    )
