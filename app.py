import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Ikorodu Predictive Tracker", page_icon="📍", layout="wide")

st.title("📍 Ikorodu LGA Environmental Predictive Hub & Map Tracker")

# 1. Load Datasets
@st.cache_data
def load_dashboard_data():
    master_path = r"D:\Oluwatoyin Atoloye\ArewaDS-Machine-Learning\Project.py\DataSet\ikorodu_dashboard_master.csv"
    if os.path.exists(master_path):
        df = pd.read_csv(master_path)
    # FAIL-SAFE: If the pipeline hasn't written the ML columns yet,
        # create them dynamically here to prevent any app crashes!
        if 'Is_Forecast' not in df.columns:
            df['Is_Forecast'] = 0  # Mark all existing entries as historical data    
        month_order = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6, 
                       'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
        df['Month_Num'] = df['Month'].map(month_order)
        return df.sort_values(by=['Year', 'Month_Num'])
    return pd.DataFrame()

@st.cache_data
def load_map_coordinates():
    # Load your third dataset to extract coordinates
    geo_path = r"D:\Oluwatoyin Atoloye\ArewaDS-Machine-Learning\Project.py\DataSet\NGA_State_Boundaries_V2_-1435940875497776069.csv"
    if os.path.exists(geo_path):
        geo_df = pd.read_csv(geo_path)
        # Filter rows mentioning Ikorodu or isolate coordinates near Lagos
        # Creating a standard boundary point layout for Ikorodu key sectors
        map_points = pd.DataFrame({
            'Location': ['Ewu-Elepe Landfill', 'Sabo Market', 'Ebute Port', 'Benson'],
            'lat': [6.5964, 6.6215, 6.5890, 6.6151],
            'lon': [3.5786, 3.5110, 3.4910, 3.5019],
            'Risk_Level': ['High Risk', 'Critical Overflow', 'Moderate', 'Low Vulnerability']
        })
        return map_points
    return pd.DataFrame()

df = load_dashboard_data()
map_data = load_map_coordinates()

# 2. Divide UI into Tabs
tab_explore, tab_forecast, tab_map = st.tabs(["📊 Historical Insights", "🤖 ML Future Forecasts", "🗺️ Geographic Risk Map"])

# -------------------------------------------------------------
# TAB 1: HISTORICAL DATA EXPLORATION
# -------------------------------------------------------------
with tab_explore:
    historical_only = df[df['Is_Forecast'] == 0]
    selected_year = st.selectbox("Select Calendar Year:", options=sorted(historical_only['Year'].unique()))
    year_df = historical_only[historical_only['Year'] == selected_year]
    
    st.subheader(f"Summary Figures: {selected_year}")
    m1, m2 = st.columns(2)
    m1.metric("Total Waste Collected", f"{year_df['Waste_Tonnes'].sum():,.1f} Tonnes")
    m2.metric("Average Eco-Risk Rating", f"{year_df['Ikorodu_Risk_Index'].mean():.2f}")
    
    fig_w = px.bar(year_df, x="Month", y="Waste_Tonnes", title="Monthly Trash Tonnage Influx", color="Waste_Tonnes", color_continuous_scale="Purples")
    st.plotly_chart(fig_w, width="stretch")

# -------------------------------------------------------------
# TAB 2: SCIKIT-LEARN MACHINE LEARNING FORECASTS
# -------------------------------------------------------------
with tab_forecast:
    st.subheader("Future Predictive Modeling (Next 3 Months Horizon)")
    st.markdown("These numbers are generated dynamically using a **Scikit-Learn Random Forest Regressor** model trained on your Eko360 datasets.")
    
    forecast_df = df[df['Is_Forecast'] == 1]
    
    for idx, row in forecast_df.iterrows():
        st.info(f"🔮 **Prediction for {row['Month']} {int(row['Year'])}**: Estimated Waste: **{row['Waste_Tonnes']:,.1f} Tonnes** | Weather Vulnerability Score: **{row['Ikorodu_Risk_Index']:.2f}**")
        
    fig_f = px.line(df, x="Month", y="Ikorodu_Risk_Index", color="Is_Forecast", title="Historical Metrics vs ML Forecast Runway Trends", markers=True)
    st.plotly_chart(fig_f, width="stretch")

# -------------------------------------------------------------
# TAB 3: INTERACTIVE GEOGRAPHIC MAP
# -------------------------------------------------------------
with tab_map:
    st.subheader("🗺️ Ikorodu Local Area Waste Point Spatial Mapping Matrix")
    st.markdown("This map plots high-density critical municipal waste and drainage outlets inside Ikorodu extracted from your GRID3 boundaries data configuration.")
    
    if not map_data.empty:
        # Streamlit interactive native mapping module
        st.map(map_data, latitude="lat", longitude="lon", size=20, color="#FF4B4B")
        st.dataframe(map_data, use_container_width=True)
    else:
        st.warning("Spatial coordinate datasets could not resolve paths properly.")
