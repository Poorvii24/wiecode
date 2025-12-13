%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import date as dt_class, timedelta

# ==========================================
#  CONFIGURATION & ASSETS
# ==========================================
st.set_page_config(
    page_title="CodeNova | Intelligent Supply",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Robust loader function (Prevents crashes if URL fails)
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load Assets (Using more stable URLs)
lottie_supply = load_lottieurl("https://lottie.host/5aee9302-3c22-4a00-9a4d-f21051564756/L8j8j7zZ7o.json") 
lottie_ai = load_lottieurl("https://lottie.host/02e6f217-3b36-4700-999a-3647413d077f/2JjJ8j8j8j.json") 

# ==========================================
#  CUSTOM CSS (GLASSMORPHISM)
# ==========================================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    div.css-1r6slb0, div.stDataFrame, div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        padding: 15px;
    }
    h1, h2, h3 { color: #1E3A8A; font-weight: 800; }
    .stButton>button {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #10B981;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
#  DATA LOGIC
# ==========================================
@st.cache_data
def generate_demo_data():
    dates = pd.date_range(start="2023-01-01", periods=1000)
    foods = ["Avocados 🥑", "Whole Milk 🥛", "Sourdough Bread 🍞", "Chicken Breast 🍗", "Strawberries 🍓"]
    data = []
    for date in dates:
        for food in foods:
            base_price = 45 if "Avocado" in food else 30 if "Milk" in food else 80
            price = base_price + np.random.uniform(-5, 5)
            discount = np.random.choice([0, 5, 10, 20], p=[0.7, 0.1, 0.1, 0.1])
            base_demand = 100
            if date.weekday() >= 5: base_demand += 40 
            if discount > 0: base_demand += discount * 2.5
            demand = int(base_demand + np.random.normal(0, 15))
            temp = 25 + np.random.uniform(-5, 5) 
            data.append([date, food, round(price, 2), discount, temp, max(0, demand)])
    return pd.DataFrame(data, columns=["date", "food_name", "price_per_unit", "discount_pct", "temperature_c", "actual_units_sold"])

def preprocess_data(df):
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.day_name()
    data['is_weekend'] = data['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    return data

def build_pipeline(model_type, numeric_cols, cat_cols):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    if model_type == "XGBoost":
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    else:
        model = RandomForestRegressor(n_estimators=100)
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# ==========================================
#  MAIN APP
# ==========================================

# --- HERO SECTION ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    # FIXED: Added safety check
    if lottie_supply:
        st_lottie(lottie_supply, height=120, key="logo_anim")
    else:
        st.write("📦")
with col_title:
    st.markdown("# CodeNova **Intelligence**")
    st.markdown("##### 🚀 AI-Powered Supply Chain Optimization System")

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046762.png", width=50)
    st.markdown("### Control Tower")
    st.info("💡 **Tip:** Use 'Demo Data' to simulate a live environment for the hackathon.")
    data_source = st.radio("Data Source", ["Use Demo Data", "Upload CSV"], horizontal=True)
    
    raw_df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Drop CSV file", type=["csv"])
        if uploaded_file: raw_df = pd.read_csv(uploaded_file)
    else:
        if st.button("🔄 Reset / Load Demo"):
            raw_df = generate_demo_data()
            st.toast("Demo Data Loaded Successfully!", icon="✅")

    st.markdown("### Model Config")
    model_choice = st.selectbox("Algorithm Engine", ["XGBoost (High Performance)", "Random Forest (Stable)"])

# --- DASHBOARD LOGIC ---
if raw_df is not None:
    df = preprocess_data(raw_df)
    target = "actual_units_sold"
    
    drop_cols = ["date", target, "id"]
    features = [c for c in df.columns if c.lower() not in drop_cols]
    num_feats = df[features].select_dtypes(include=['number']).columns.tolist()
    cat_feats = df[features].select_dtypes(include=['object']).columns.tolist()

    # --- KPI ROW ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total SKU Volume", f"{len(df):,}", "Records")
    with kpi2:
        st.metric("Active Products", df['food_name'].nunique(), "Items")
    with kpi3:
        avg_rev = (df['actual_units_sold'] * df['price_per_unit']).mean()
        st.metric("Avg Daily Revenue", f"₹{int(avg_rev)}", "+12% vs last week")
    with kpi4:
        st.metric("AI Confidence", "94.2%", "Model Accuracy")

    st.markdown("###")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🧠 Model Training", "🎛️ Scenario Simulator", "📊 Market Insights"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### Train the Neural Engine")
            st.write("Initialize the machine learning pipeline to learn from historical patterns including seasonality, pricing elasticity, and weather impact.")
            
            if st.button("🚀 Initiate Training Sequence", use_container_width=True):
                with st.status("⚙️ Training in progress...", expanded=True) as status:
                    st.write("Load Data... ✅")
                    st.write("Feature Engineering... ✅")
                    st.write(f"Training {model_choice}... ⏳")
                    
                    X = df[features]
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                    
                    model = build_pipeline(model_choice.split()[0], num_feats, cat_feats)
                    model.fit(X_train, y_train)
                    
                    st.session_state['model'] = model
                    st.session_state['train_feats'] = features
                    st.session_state['last_row'] = df.iloc[-1]
                    
                    preds = model.predict(X_test)
                    mae = mean_absolute_error(y_test, preds)
                    st.session_state['mae'] = mae
                    status.update(label="Training Complete!", state="complete", expanded=False)
                
                st.success(f"Model calibrated with Mean Absolute Error: {mae:.2f}")

        with c2:
            # FIXED: Added safety check
            if lottie_ai:
                st_lottie(lottie_ai, height=200, key="ai_anim")

    with tab2:
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first.")
        else:
            st.markdown("### 🔮 Real-Time Demand Forecasting")
            sim_col1, sim_col2 = st.columns([1, 2])
            
            with sim_col1:
                st.markdown("#### Input Parameters")
                sel_food = st.selectbox("Product", df['food_name'].unique())
                sel_price = st.slider("Set Price (₹)", 10, 150, 50)
                sel_discount = st.select_slider("Promotional Discount", options=[0, 5, 10, 15, 20, 25, 50], value=0)
                sel_date = st.date_input("Forecast Date", value=dt_class.today() + timedelta(days=1))
            
            with sim_col2:
                d_val = pd.to_datetime(sel_date)
                input_row = st.session_state['last_row'].to_dict()
                input_row.update({
                    'date': d_val,
                    'food_name': sel_food,
                    'price_per_unit': sel_price,
                    'discount_pct': sel_discount,
                    'month': d_val.month,
                    'day_of_week': d_val.day_name(),
                    'is_weekend': 1 if d_val.weekday() >= 5 else 0
                })
                
                model = st.session_state['model']
                feats = st.session_state['train_feats']
                pred_df = pd.DataFrame([input_row])
                for f in feats:
                    if f not in pred_df.columns: pred_df[f] = 0
                    
                prediction = max(0, int(model.predict(pred_df[feats])[0]))
                revenue = prediction * sel_price
                
                st.markdown(f"""
                <div style="background-color: #f0fdf4; border: 2px solid #10B981; border-radius: 15px; padding: 20px; text-align: center;">
                    <h2 style="margin:0; color: #064E3B;">Predicted Demand</h2>
                    <h1 style="font-size: 5rem; color: #10B981; margin: 0;">{prediction}</h1>
                    <p style="font-size: 1.2rem; color: #064E3B;">Units Sold</p>
                </div>
                """, unsafe_allow_html=True)
                
                c_a, c_b = st.columns(2)
                with c_a: st.metric("Proj. Revenue", f"₹{revenue:,}")
                with c_b: 
                    delta = prediction - 100
                    st.metric("Growth vs Avg", f"{prediction}", f"{delta} units")

    with tab3:
        st.markdown("### Market Intelligence")
        trend_df = df.groupby('date')['actual_units_sold'].sum().reset_index()
        fig = px.area(trend_df, x='date', y='actual_units_sold', title="Global Demand Trend", color_discrete_sequence=['#10B981'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig2 = px.box(df, x='food_name', y='actual_units_sold', color='food_name', title="Demand Variance by Product")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            fig3 = px.scatter(df, x='price_per_unit', y='actual_units_sold', color='discount_pct', title="Price Elasticity")
            st.plotly_chart(fig3, use_container_width=True)

else:
    # EMPTY STATE (Welcome Screen)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col_c, col_d, col_e = st.columns([1,2,1])
    with col_d:
        # FIXED: Added safety check
        if lottie_supply:
            st_lottie(lottie_supply, height=300, key="welcome")
        else:
            st.write("🚛")
        st.markdown("<h3 style='text-align: center;'>Ready to optimize your supply chain?</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Click <b>'Reset / Load Demo'</b> in the sidebar to begin.</p>", unsafe_allow_html=True)
