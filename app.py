import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =========================================
# Load saved model + feature columns
# =========================================
best_model = joblib.load("best_food_demand_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# =========================================
# Column configs from training pipeline
# =========================================
DATE_COL = "date"

CATEGORICAL_COLS = [
    "day_name",
    "division",
    "region_pin_prefix",
    "time_slot",
    "festival_name",
    "holiday_type",
    "food_category",
    "food_subcategory",
    "food_name",
    "regional_specialty",
]

NUMERIC_COLS = [
    "week_number",
    "day_of_week",
    "festival_flag",
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "price_per_unit",
    "discount_pct",
    "historical_demand_lag1",
    "historical_demand_lag7",
    "past_sales_avg_30d",
    "competitor_density",
    "population_density",
    "inflation_rate_monthly",
    "restaurant_capacity",
]

# =========================================
# Preprocessing for single prediction input
# =========================================
def preprocess_input(raw_input: dict):
    df = pd.DataFrame([raw_input])

    # Convert date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["week_number"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df[DATE_COL].dt.weekday
    df["day_name"] = df[DATE_COL].dt.day_name()

    # More date features used during training
    df["dayofweek_num"] = df[DATE_COL].dt.dayofweek
    df["month"] = df[DATE_COL].dt.month

    # Numeric cleaning
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Align with training columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df.reindex(columns=feature_cols, fill_value=0)

    return df


# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="Food Demand Forecasting", layout="centered")
st.title("🍛 Food Demand Forecasting – Karnataka")
st.caption("Predict food demand using your trained ML model")

st.markdown("---")
st.header("📝 Enter Input Details")

# ----------------------------
# Input Form
# ----------------------------
with st.form("input_form"):
    date_val = st.date_input("Date", value=datetime.today())

    division = st.text_input("Division", "Bengaluru")
    region_pin_prefix = st.text_input("Region PIN Prefix", "560")
    time_slot = st.selectbox("Time Slot", ["Breakfast", "Lunch", "Evening", "Dinner"])

    festival_flag = st.selectbox("Festival Day?", [0, 1])
    festival_name = st.text_input("Festival Name", "None")
    holiday_type = st.selectbox("Holiday Type", ["None", "Public Holiday", "Weekend", "Religious"])

    temperature_c = st.number_input("Temperature (°C)", value=28.0)
    humidity_pct = st.number_input("Humidity (%)", value=60.0)
    rainfall_mm = st.number_input("Rainfall (mm)", value=0.0)

    food_category = st.text_input("Food Category", "North Indian")
    food_subcategory = st.text_input("Food Subcategory", "Thali")
    food_name = st.text_input("Food Name", "Veg Thali")
    regional_specialty = st.selectbox("Regional Specialty?", ["Yes", "No"])

    price_per_unit = st.number_input("Price per Unit (₹)", value=150.0)
    discount_pct = st.number_input("Discount (%)", value=0.0)

    historical_demand_lag1 = st.number_input("Yesterday's Demand", value=50.0)
    historical_demand_lag7 = st.number_input("Demand Last Week", value=60.0)
    past_sales_avg_30d = st.number_input("30-Day Avg Demand", value=55.0)

    competitor_density = st.number_input("Competitor Density (0–1)", value=0.5)
    population_density = st.number_input("Population Density (0–1)", value=0.6)
    inflation_rate_monthly = st.number_input("Inflation Rate (%)", value=5.0)
    restaurant_capacity = st.number_input("Restaurant Capacity", value=80.0)

    submitted = st.form_submit_button("Predict Demand")

# ----------------------------
# Prediction Logic
# ----------------------------
if submitted:

    input_dict = {
        DATE_COL: date_val,
        "division": division,
        "region_pin_prefix": region_pin_prefix,
        "time_slot": time_slot,
        "festival_flag": festival_flag,
        "festival_name": festival_name,
        "holiday_type": holiday_type,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "rainfall_mm": rainfall_mm,
        "food_category": food_category,
        "food_subcategory": food_subcategory,
        "food_name": food_name,
        "regional_specialty": regional_specialty,
        "price_per_unit": price_per_unit,
        "discount_pct": discount_pct,
        "historical_demand_lag1": historical_demand_lag1,
        "historical_demand_lag7": historical_demand_lag7,
        "past_sales_avg_30d": past_sales_avg_30d,
        "competitor_density": competitor_density,
        "population_density": population_density,
        "inflation_rate_monthly": inflation_rate_monthly,
        "restaurant_capacity": restaurant_capacity,
    }

    X = preprocess_input(input_dict)
    prediction = best_model.predict(X)[0]

    # Add buffer for optimisation
    recommended_units = int(np.ceil(prediction * 1.10))

    st.markdown("---")
    st.header("📌 Prediction Result")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Demand", f"{prediction:.2f} units")
    col2.metric("Recommended Units (10% buffer)", recommended_units)

    st.success("Prediction generated using your trained model 🎉")
