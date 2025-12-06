# 🍛 Food Demand Forecasting – Karnataka  
### Machine Learning + Streamlit App | Hackathon Project

This project predicts **daily food demand** across regions in Karnataka using  
historical sales, weather conditions, pricing signals, festival patterns, and demographic factors.

It was developed for a **12-hour hackathon** and includes a complete  
ML pipeline + a production-ready Streamlit web application.

---

## 🚀 Features

### 🔹 Machine Learning  
- Multiple regression models trained (Linear Regression, Random Forest, XGBoost)  
- Automatic model comparison (MAE / RMSE)  
- XGBoost selected as the best-performing model  
- Feature engineering: date-time extraction, lag variables, categorical encoding  

### 🔹 Streamlit UI  
- User enters features manually (date, pricing, weather, food category, etc.)  
- Model predicts:  
  ✔ **expected demand (units)**  
  ✔ **recommended optimized units** (with buffer)  
- Lightweight, clean, responsive UI  
- Suitable for real-time forecasting or business demo  

### 🔹 Deployment Ready  
- Includes `requirements.txt`  
- Works on:  
  ✔ Streamlit Cloud  
  ✔ Local machine  
  ✔ Google Colab (via Cloudflare tunnel)

---

## 📂 Folder Structure

