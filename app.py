%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import date as dt_class, timedelta

# ==========================================
#  UI & CSS CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="CodeNova | AI Supply Chain",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Dashboard" look
st.markdown("""
    <style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32; /* Green */
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #2E7D32;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
#  HELPER FUNCTIONS
# ==========================================
@st.cache_data
def generate_demo_data():
    """Generates realistic dummy data for the demo"""
    dates = pd.date_range(start="2023-01-01", periods=1000)
    foods = ["Organic Apples", "Whole Milk", "Avocados", "Fresh Bread", "Chicken Breast"]
    
    data = []
    for date in dates:
        # Simulate seasonality and random demand
        for food in foods:
            base_price = 50 if food == "Organic Apples" else 30 if food == "Whole Milk" else 80
            price = base_price + np.random.uniform(-5, 5)
            discount = np.random.choice([0, 5, 10, 20], p=[0.7, 0.1, 0.1, 0.1])
            
            # Demand logic (fake but realistic)
            base_demand = 100
            if date.weekday() >= 5: base_demand += 30 # Weekends high
            if discount > 0: base_demand += discount * 2 # Discounts boost sales
            
            demand = int(base_demand + np.random.normal(0, 15))
            
            data.append([date, food, round(price, 2), discount, max(0, demand)])
            
    return pd.DataFrame(data, columns=["date", "food_name", "price_per_unit", "discount_pct", "actual_units_sold"])

def preprocess_data(df):
    data = df.copy()
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.day_name()
        data['is_weekend'] = data['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    return data

def build_pipeline(model_type, numeric_cols, cat_cols, params):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    if model_type == "XGBoost":
        model = XGBRegressor(n_estimators=params['n_est'], learning_rate=params['lr'], max_depth=params['depth'], random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# ==========================================
#  SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=80)
    st.markdown("## **CodeNova AI**")
    st.markdown("Dynamic Demand Forecasting System")
    st.markdown("---")
    
    st.write("### 📂 Data Source")
    data_source = st.radio("Choose source:", ["Upload CSV", "Use Demo Data"])
    
    raw_df = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file: raw_df = pd.read_csv(uploaded_file)
    else:
        if st.button("🔄 Load Demo Data"):
            raw_df = generate_demo_data()
            st.success("Demo data loaded!")

    st.markdown("---")
    st.write("### ⚙️ Model Config")
    model_choice = st.selectbox("Algorithm", ["XGBoost", "Random Forest"])
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)

# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.markdown('<div class="main-header">CodeNova Intelligent Supply</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Reducing food waste through AI-driven demand prediction.</div>', unsafe_allow_html=True)

if raw_df is not None:
    # Preprocessing
    df = preprocess_data(raw_df)
    target = "actual_units_sold"
    
    # Feature Engineering
    features = [c for c in df.columns if c not in ["date", target, "id"]]
    num_feats = df[features].select_dtypes(include=['number']).columns.tolist()
    cat_feats = df[features].select_dtypes(include=['object']).columns.tolist()

    # --- TOP METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Historical Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['food_name'].nunique()}</div>
            <div class="metric-label">Unique Products</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_sales = int(df[target].mean())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_sales}</div>
            <div class="metric-label">Avg Daily Sales</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">94%</div>
            <div class="metric-label">Target Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TABS ---
    tab_train, tab_pred, tab_viz = st.tabs(["🚀 Model Training", "🔮 Live Prediction", "📈 Analytics"])

    # 1. TRAINING TAB
    with tab_train:
        st.subheader("Train the AI Model")
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.info("The system uses historical sales, pricing, and calendar data to learn demand patterns.")
            if st.button("🚀 Start Training Pipeline", use_container_width=True):
                with st.spinner("Training XGBoost Regressor..."):
                    # Train/Test Split
                    X = df[features]
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                    
                    # Model Pipeline
                    params = {'n_est': 100, 'lr': learning_rate, 'depth': 6}
                    model = build_pipeline(model_choice, num_feats, cat_feats, params)
                    model.fit(X_train, y_train)
                    
                    # Store in session
                    st.session_state['model'] = model
                    st.session_state['ref_df'] = df
                    st.session_state['features'] = (num_feats, cat_feats)
                    
                    # Eval
                    preds = model.predict(X_test)
                    mae = mean_absolute_error(y_test, preds)
                    st.session_state['mae'] = mae
                    st.session_state['y_test'] = y_test
                    st.session_state['preds'] = preds
                    
                st.success(f"Training Complete! Model MAE: {mae:.2f}")

        with col_t2:
            if 'preds' in st.session_state:
                # Stylish Chart
                res_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['preds']})
                fig = px.scatter(res_df, x='Actual', y='Predicted', 
                                 title="Model Accuracy (Actual vs Predicted)",
                                 color_discrete_sequence=['#2E7D32'],
                                 template="plotly_white")
                fig.add_shape(type="line", line=dict(dash='dash'),
                            x0=res_df['Actual'].min(), y0=res_df['Actual'].min(),
                            x1=res_df['Actual'].max(), y1=res_df['Actual'].max())
                st.plotly_chart(fig, use_container_width=True)

    # 2. PREDICTION TAB
    with tab_pred:
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model in the previous tab first.")
        else:
            st.subheader("Simulation Dashboard")
            
            col_input, col_result = st.columns([1, 2])
            
            with col_input:
                st.markdown("#### 🎛️ Control Panel")
                food_item = st.selectbox("Select Product", df['food_name'].unique())
                pred_date = st.date_input("Forecast Date", value=dt_class.today() + timedelta(days=1))
                
                # Get stats for defaults
                item_stats = df[df['food_name'] == food_item]
                avg_price = item_stats['price_per_unit'].mean()
                
                price = st.slider("Unit Price (₹)", 10.0, 200.0, float(avg_price))
                discount = st.radio("Apply Discount?", [0, 5, 10, 20], horizontal=True, format_func=lambda x: f"{x}%")
            
            with col_result:
                if st.button("⚡ Generate Forecast", use_container_width=True):
                    # Construct Input
                    d_val = pd.to_datetime(pred_date)
                    input_data = pd.DataFrame({
                        'date': [d_val],
                        'food_name': [food_item],
                        'price_per_unit': [price],
                        'discount_pct': [discount],
                        'month': [d_val.month],
                        'day_of_week': [d_val.day_name()],
                        'is_weekend': [1 if d_val.weekday() >= 5 else 0]
                    })
                    
                    model = st.session_state['model']
                    prediction = model.predict(input_data)[0]
                    
                    # Display Big Metrics
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <h3 style="color: #555; margin:0;">Forecasted Demand</h3>
                            <h1 style="color: #2E7D32; font-size: 3.5rem; margin:0;">{int(prediction)}</h1>
                            <p>Units</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                         st.markdown(f"""
                        <div style="text-align: center;">
                            <h3 style="color: #555; margin:0;">Est. Revenue</h3>
                            <h1 style="color: #1976D2; font-size: 3.5rem; margin:0;">₹{int(prediction * price)}</h1>
                            <p>INR</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        stock_status = "✅ Sufficient Stock" if prediction < 150 else "⚠️ Low Stock Alert"
                        color = "green" if prediction < 150 else "red"
                        st.markdown(f"""
                        <div style="text-align: center; padding-top: 20px;">
                            <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px;">
                                {stock_status}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # 3. ANALYTICS TAB
    with tab_viz:
        st.subheader("Market Trends")
        if df is not None:
            c1, c2 = st.columns(2)
            with c1:
                trend_data = df.groupby("date")[target].sum().reset_index()
                fig_trend = px.line(trend_data, x='date', y=target, title="Total Demand Trend", template="plotly_white", line_shape='spline')
                fig_trend.update_traces(line_color='#2E7D32', line_width=3)
                st.plotly_chart(fig_trend, use_container_width=True)
            with c2:
                fig_bar = px.bar(df, x='food_name', y=target, color='food_name', title="Sales Distribution by Product", template="plotly_white")
                st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("👈 Please upload a CSV file or click 'Load Demo Data' in the sidebar to begin.")
