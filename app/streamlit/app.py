import streamlit as st
import requests
import base64

st.title("Freight Cost Estimator")

# Top-level layout: inputs and graph side-by-side
left_col, right_col = st.columns([2, 3])

# Top-level column pair for site + commodity
col1, col2 = st.columns(2)
site = col1.selectbox("Select Site", ["DIT", "SPW", "SPN", "SPCP", "SPT",
                                      "PVF", "SPHU", "SPTM", "FSU", "CTS", "SPJ"])
commodity = col2.selectbox("Select Commodity", ["1CBL", "1VNL", "1CPT"])

# Top-level column pair for quantity + unit of measure
col3, col4 = st.columns(2)
quantity = col3.number_input("Enter Quantity", min_value=0.0, step=1.0)
uom = col4.selectbox("Select Unit of Measure", ["SQYD", "LBS"])

# Button and API call
if st.button("Estimate Freight"):
    payload = {
        "site": site,
        "commodity": commodity,
        "quantity": quantity,
        "uom": uom
    }
    response = requests.post(
        "http://localhost:8000/api/estimate", json=payload)
    if response.status_code == 200:
        st.session_state.result = response.json()
    else:
        try:
            error_message = response.json().get("detail", "Unknown error")
        except Exception:
            error_message = "Failed to get estimate"
        st.error(f"❌ Error: {error_message}")


# Show results
if "result" in st.session_state:
    result = st.session_state.result

    with left_col:
        st.write(f"**LTL Cost:** ${result['ltl_cost']:.2f}")
        st.write(f"**FTL Cost:** ${result['ftl_cost']:.2f}")
        st.write(f"**Optimal Mode:** {result['optimal_mode']}")
        st.write(f"**LTL Rate:** {result['ltl_rate']}")
        st.write(f"**Freight Class:** {result['freight_class']}")

    with right_col:
        img_data = base64.b64decode(result["plot_base64"])
        st.image(img_data, caption="Cost Curve Plot", use_column_width=True)
