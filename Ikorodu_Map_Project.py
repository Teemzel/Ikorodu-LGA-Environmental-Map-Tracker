import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# 1. PARSE SATELLITE RAINFALL DATA FOR IKORODU
# =====================================================================
def clean_ikorodu_rainfall(file_path):
    print("🌧️ Processing NASA IMERG satellite rainfall data...")
    df = pd.read_csv("D:\\Oluwatoyin Atoloye\\ArewaDS-Machine-Learning\\Project.py\\DataSet\\Lagos Precipitation.csv")
    
    if 'file_name' in df.columns:
        df = df[df['file_name'].str.contains('V06', na=False, case=False)]
        
    ikorodu_spatial = df[
        (df['lat'].between(6.55, 6.70)) & 
        (df['lon'].between(3.40, 3.65))
    ].copy()
    
    if ikorodu_spatial.empty:
        ikorodu_spatial = df.copy()

    base_date = pd.to_datetime('1970-01-01')
    ikorodu_spatial['Calendar_Date'] = base_date + pd.to_timedelta(ikorodu_spatial['date'], unit='D')
    ikorodu_spatial['Month'] = ikorodu_spatial['Calendar_Date'].dt.strftime('%b').str.upper()
    
    monthly_rain_baseline = ikorodu_spatial.groupby(['Month'], as_index=False)['precipitation'].mean()
    monthly_rain_baseline.rename(columns={'precipitation': 'Rainfall_Volume_mm'}, inplace=True)
    return monthly_rain_baseline

# =====================================================================
# 2. AGGRESSIVE CELL CONVERTER FOR EXCEL WASTE DATA (IKORODU)
# =====================================================================
def parse_ikorodu_excel_waste(file_path):
    print("♻️ Scanning Eko360 Excel columns aggressively...")
    df = pd.read_excel("D:\\Oluwatoyin Atoloye\\ArewaDS-Machine-Learning\\Project.py\\DataSet\\Statistics of refused deposited at various Landfill sites for the year 2022 - 2024__.xlsx", sheet_name=0, header=None)
    
    extracted_records = []
    current_year = None
    
    valid_months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    
    for index, row in df.iterrows():
        first_cell = str(row.iloc[0]).strip().upper()
        
        # 1. Find the Year blocks
        if "FOR THE Y202" in first_cell or "YEAR 202" in first_cell or "Y202" in first_cell:
            for word in first_cell.split():
                if "202" in word:
                    year_digits = ''.join(filter(str.isdigit, word))
                    if len(year_digits) == 4:
                        current_year = int(year_digits)
                        print(f"📍 Scanner locked onto Data Year: {current_year}")
            continue
            
        # 2. Extract Month Rows
        month_prefix = first_cell[:3]
        if current_year and month_prefix in valid_months and "TOTAL" not in first_cell:
            try:
                # Target Column index 10 (Column K - Ikorodu Waste Tonnage)
                raw_waste_value = row.iloc[10]
                
                # FORCE CONVERSION: Clean out text strings, commas, or spaces from the cell value
                if pd.notna(raw_waste_value):
                    clean_string = str(raw_waste_value).replace(',', '').replace(' ', '').strip()
                    numeric_waste = float(clean_string)
                    
                    extracted_records.append({
                        'Year': int(current_year),
                        'Month': month_prefix,
                        'Waste_Tonnes': numeric_waste
                    })
            except (ValueError, IndexError, TypeError):
                # If a specific cell conversion hits text, skip it safely
                continue
                
    clean_df = pd.DataFrame(extracted_records)
    return clean_df

# =====================================================================
# 3. MACHINE LEARNING FORECASTING ENGINE (SCIKIT-LEARN)
# =====================================================================
def generate_ml_forecasts(historical_df, rain_baseline):
    print("🤖 Running Random Forest Forecasting Engine...")
    
    month_map = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6, 
                 'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
    
    historical_df['Month_Num'] = historical_df['Month'].map(month_map)
    
    X = historical_df[['Year', 'Month_Num']]
    y = historical_df['Waste_Tonnes']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict the next 3 sequential months in the timeline
    last_row = historical_df.sort_values(by=['Year', 'Month_Num']).iloc[-1]
    future_records = []
    curr_y = int(last_row['Year'])
    curr_m = int(last_row['Month_Num'])
    
    for _ in range(3):
        curr_m += 1
        if curr_m > 12:
            curr_m = 1
            curr_y += 1
        future_records.append({'Year': curr_y, 'Month_Num': curr_m, 'Is_Forecast': 1})
        
    future_df = pd.DataFrame(future_records)
    future_df['Waste_Tonnes'] = model.predict(future_df[['Year', 'Month_Num']])
    
    rev_month_map = {v: k for k, v in month_map.items()}
    future_df['Month'] = future_df['Month_Num'].map(rev_month_map)
    
    future_df = pd.merge(future_df, rain_baseline, on='Month', how='left')
    future_df['Ikorodu_Risk_Index'] = (future_df['Waste_Tonnes'] * (future_df['Rainfall_Volume_mm'] + 1)) / 1000
    
    historical_df['Is_Forecast'] = 0
    
    combined_output = pd.concat([historical_df, future_df], ignore_index=True)
    return combined_output.drop(columns=['Month_Num'])

# =====================================================================
# 4. CONTROL LOOP EXECUTION
# =====================================================================
def run_pipeline():
    print("🚀 Initializing Dynamic Predictive Data Pipeline...")
    
    waste_path = r"D:\Oluwatoyin Atoloye\ArewaDS-Machine-Learning\Project.py\DataSet\Statistics of refused deposited at various Landfill sites for the year 2022 - 2024__.xlsx"
    rain_path = r"D:\Oluwatoyin Atoloye\ArewaDS-Machine-Learning\Project.py\DataSet\Lagos Precipitation.csv"
    
    rain_data = clean_ikorodu_rainfall(rain_path)
    waste_data = parse_ikorodu_excel_waste(waste_path)
    
    if waste_data.empty:
        print("⚠️ Data Parse Error: The data conversion layer could not extract any numeric row metrics from column index 10.")
        return
        
    waste_data['Month'] = waste_data['Month'].astype(str).str.strip().str.upper()
    rain_data['Month'] = rain_data['Month'].astype(str).str.strip().str.upper()
    
    merged_df = pd.merge(waste_data, rain_data, on='Month', how='left')
    merged_df['Rainfall_Volume_mm'] = merged_df['Rainfall_Volume_mm'].fillna(0.0)
    merged_df['Ikorodu_Risk_Index'] = (merged_df['Waste_Tonnes'] * (merged_df['Rainfall_Volume_mm'] + 1)) / 1000
    
    final_df = generate_ml_forecasts(merged_df, rain_data)
    
    output_destination = r"D:\Oluwatoyin Atoloye\ArewaDS-Machine-Learning\Project.py\DataSet\ikorodu_dashboard_master.csv"
    final_df.to_csv(output_destination, index=False)
    
    print(f"\n🎉 SUCCESS! Generated {len(final_df)} structured data profile entries with ML Forecast rows.")

if __name__ == "__main__":
    run_pipeline()
