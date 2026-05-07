import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Smart Energy Analytics",
    page_icon="⚡",
    layout="wide"
)

# =========================================
# LOAD DATA
# =========================================

base_path = "/home/hp/bigdata-project/uts-tbg-230104040206/output"

energy_total = pd.read_parquet(
    f"{base_path}/energy_total"
)

energy_time = pd.read_parquet(
    f"{base_path}/energy_time"
)

ml_energy = pd.read_parquet(
    f"{base_path}/ml_energy"
)

# =========================================
# HEADER
# =========================================

st.title("⚡ Smart Energy Consumption Analytics")

st.markdown("""
Dashboard Big Data untuk monitoring konsumsi energi 
berbasis PySpark, Parquet, Machine Learning, dan Streamlit.
""")

st.divider()

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("⚙️ Filter Dashboard")

sector_option = st.sidebar.selectbox(
    "Pilih Sektor",
    energy_time["sector"].unique()
)

filtered = energy_time[
    energy_time["sector"] == sector_option
]

# =========================================
# KPI SECTION
# =========================================

total_konsumsi = filtered["avg_power_usage"].sum()

rata_konsumsi = filtered["avg_power_usage"].mean()

maks_konsumsi = filtered["avg_power_usage"].max()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "⚡ Total Konsumsi",
        f"{total_konsumsi:.2f} kWh"
    )

with col2:
    st.metric(
        "📈 Rata-rata Konsumsi",
        f"{rata_konsumsi:.2f} kWh"
    )

with col3:
    st.metric(
        "🔥 Konsumsi Tertinggi",
        f"{maks_konsumsi:.2f} kWh"
    )

st.divider()

# =========================================
# CHART
# =========================================

st.subheader(f"📊 Tren Konsumsi Energi - {sector_option}")

fig = px.line(
    filtered,
    x="hour",
    y="avg_power_usage",
    markers=True,
    line_shape="spline",
    title=f"Energy Trend {sector_option}"
)

fig.update_layout(
    xaxis_title="Jam",
    yaxis_title="Average Power Usage (kWh)",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# =========================================
# MACHINE LEARNING
# =========================================

st.divider()

st.subheader("🤖 AI Energy Prediction")

X = ml_energy[["hour"]]

y = ml_energy["power_usage"]

model = LinearRegression()

model.fit(X, y)

future_hour = st.slider(
    "Pilih Jam Prediksi",
    0,
    23,
    12
)

prediction = model.predict([[future_hour]])

st.success(
    f"Prediksi konsumsi energi pada jam "
    f"{future_hour}: "
    f"{prediction[0]:.2f} kWh"
)

# =========================================
# FOOTER
# =========================================

st.divider()

st.caption(
    "UTS Teknologi Big Data • TI23A • Smart Energy Consumption Analytics"
)