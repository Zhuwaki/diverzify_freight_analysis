import streamlit as st
import requests
import pandas as pd
import os

# or set this in Streamlit Cloud
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Freight Estimator", layout="wide")
st.title("📦 Freight Estimator Tool")

st.markdown(
    "Upload your batch freight input file and estimate dual freight costs (CWT + Area)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully")

    if st.button("Estimate Freight Costs"):
        with st.spinner("⏳ Processing..."):
            try:
                files = {"file": (uploaded_file.name,
                                  uploaded_file.getvalue(), "text/csv")}
                response = requests.post(f"{API_URL}/batch", files=files)
                data = response.json()

                if "preview" in data:
                    st.success(f"✅ Processed {data['rows_processed']} rows")

                    st.subheader("📋 Preview Results")
                    preview_df = pd.DataFrame(data["preview"])
                    st.dataframe(preview_df, use_container_width=True)

                    st.markdown("---")
                    st.markdown(
                        f"[📥 Click here to download the full result file]({API_URL}{data['download_url']})")
                else:
                    st.error(
                        f"❌ {data.get('error', 'Unknown error occurred')}")

            except Exception as e:
                st.error(f"💥 Exception: {str(e)}")
else:
    st.info("📁 Please upload a .csv file to begin.")
