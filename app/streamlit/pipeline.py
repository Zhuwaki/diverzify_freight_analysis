import streamlit as st
import requests

# 📍 Update if deployed elsewhere
API_URL = "http://localhost:8000/api/pipeline"

st.set_page_config(page_title="Freight API Pipeline", layout="wide")
st.title("🚛 Freight Estimation Pipeline")

st.markdown(
    """
    Upload your raw freight input file (CSV or Excel) and let the API run the full pipeline:
    - Cleaning
    - Model Estimation
    - Final Cost Comparison
    """
)

uploaded_file = st.file_uploader(
    "📤 Upload your input file", type=["csv", "xlsx"])

if uploaded_file:
    with st.spinner("⏳ Processing pipeline... this may take a few seconds"):
        res = requests.post(
            API_URL,
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }
        )

    if res.status_code == 200:
        data = res.json()
        st.success("✅ Pipeline complete!")

        st.subheader("📥 Download Outputs")
        st.markdown(
            f"[🧼 Cleaned File]({data['clean_download']})", unsafe_allow_html=True)
        st.markdown(
            f"[📊 Modeled File]({data['model_download']})", unsafe_allow_html=True)
        st.markdown(
            f"[📈 Final Comparison File]({data['final_download']})", unsafe_allow_html=True)

        st.subheader("🔍 Final Preview")
        st.dataframe(data["final_preview"])

    else:
        st.error("❌ Pipeline failed")
        try:
            st.json(res.json())
        except Exception:
            st.write(res.text)
