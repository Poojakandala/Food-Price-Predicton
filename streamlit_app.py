import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Food Price Prediction", layout="wide")

# --------------------------------------------------
# Background Styling
# --------------------------------------------------
st.markdown("""
<style>

/* Full page background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1498837167922-ddd27525d352");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.85);  /* DARKER overlay */
    z-index: -1;
}

/* Make ALL text white */
html, body, [class*="css"]  {
    color: white !important;
}

/* Main container styling */
.block-container {
    background-color: rgba(0, 0, 0, 0.6);
    padding: 2rem;
    border-radius: 20px;
}

/* Make title big and visible */
h1 {
    color: #FFD700 !important;  /* Gold color */
    font-size: 42px !important;
    text-align: center;
}

/* Subheaders */
h2, h3 {
    color: #00FFCC !important;
}

</style>
""", unsafe_allow_html=True)
st.title("🍽 Food Checkout Price Prediction System")

# --------------------------------------------------
# Load Dataset Directly
# --------------------------------------------------
df = pd.read_csv("E:/pooja/food-demand/food_Demand_test.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Define X and Y
X = df.drop(["id", "checkout_price"], axis=1)

# Apply log transformation
y = np.log1p(df["checkout_price"])  # log(1 + price)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Model Selection
# --------------------------------------------------
st.subheader("🤖 Select Regression Technique")

model_name = st.selectbox(
    "Choose Model",
    ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"]
)

if model_name == "Linear Regression":
    model = LinearRegression()

elif model_name == "Decision Tree":
    model = DecisionTreeRegressor()

elif model_name == "Random Forest":
    model = RandomForestRegressor(n_estimators=200, random_state=42)

elif model_name == "XGBoost":
    model = XGBRegressor(n_estimators=300, learning_rate=0.05)

elif model_name == "LightGBM":
    model = LGBMRegressor(n_estimators=300, learning_rate=0.05)

# Train Model
model.fit(X_train, y_train)
# Predict (log scale)
y_pred_log = model.predict(X_test)

# Convert back to original scale
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
st.subheader("📈 Model Performance")

mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", round(mae, 2))
col2.metric("RMSE", round(rmse, 2))
col3.metric("R² Score", round(r2, 4))


# --------------------------------------------------
# 3D Visualization
# --------------------------------------------------
st.subheader("🌐 3D Visualization")

fig_3d = px.scatter_3d(
    df,
    x="base_price",
    y="week",
    z="checkout_price",
    color="homepage_featured",
    title="3D View: Base Price vs Week vs Checkout Price"
)

st.plotly_chart(fig_3d, use_container_width=True)


# --------------------------------------------------
# Pie Chart Visualization
# --------------------------------------------------
st.subheader("🥧 Promotion Distribution")

pie_data = df["emailer_for_promotion"].value_counts().reset_index()
pie_data.columns = ["Promotion", "Count"]

fig_pie = px.pie(
    pie_data,
    names="Promotion",
    values="Count",
    title="Emailer Promotion Distribution"
)

st.plotly_chart(fig_pie, use_container_width=True)


# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.subheader("🔮 Predict New Checkout Price")

week = st.number_input("Week", min_value=1)
center_id = st.number_input("Center ID", min_value=1)
meal_id = st.number_input("Meal ID", min_value=1)
base_price = st.number_input("Base Price", min_value=0.0)
emailer = st.selectbox("Emailer Promotion", [0, 1])
homepage = st.selectbox("Homepage Featured", [0, 1])

if st.button("Predict Price"):
    
    input_data = pd.DataFrame([[
        week,
        center_id,
        meal_id,
        base_price,
        emailer,
        homepage
    ]], columns=[
        "week",
        "center_id",
        "meal_id",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured"
    ])
    
    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)

    # Safety check
    prediction = max(0, prediction)

    st.success(f"💰 Predicted Checkout Price: ₹ {round(prediction, 2)}")