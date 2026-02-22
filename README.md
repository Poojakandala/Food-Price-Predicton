# 🍽 Food Price Prediction

A Machine Learning web application built with **Streamlit** to predict food checkout prices using multiple regression algorithms.

---

## 📌 Project Overview

This project predicts the **checkout price of food items** using historical food demand data.

The application allows users to:

- Select different regression models
- View model performance metrics (MAE, RMSE, R²)
- Visualize data using 3D plots
- View promotion distribution via pie chart
- Predict checkout price for new inputs

---

## 🚀 Features

### 🤖 Multiple Regression Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

### 📊 Data Visualization
- 3D Scatter Plot (Base Price vs Week vs Checkout Price)
- Promotion Distribution Pie Chart

### 📈 Model Evaluation
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### 🎨 User Interface
- Beautiful background styling
- Model selection dropdown
- Real-time prediction system

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Plotly

---

## 📊 Dataset

Food Demand Forecasting dataset containing:

| Column Name              | Description |
|--------------------------|------------|
| week                     | Week number |
| center_id                | Distribution center ID |
| meal_id                  | Meal ID |
| base_price               | Base price of the meal |
| emailer_for_promotion    | Email promotion flag (0/1) |
| homepage_featured        | Homepage featured flag (0/1) |
| checkout_price           | Final checkout price (Target Variable) |

---

## 📂 Project Structure

```
Food-Price-Prediction/
│
├── app.py
├── food_Demand_test.csv
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Food-Price-Prediction.git
cd Food-Price-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm plotly
```

---

## ▶ Run the Application

```bash
streamlit run app.py
```

The app will run locally at:

```
http://localhost:8501
```

---

## 🔮 How Prediction Works

1. Dataset is loaded directly.
2. Log transformation is applied on `checkout_price`.
3. Selected regression model is trained.
4. Predictions are converted back to original scale using exponential transformation.
5. Final predicted checkout price is displayed.

---

## 📌 Future Improvements

- Hyperparameter tuning
- Feature engineering
- Model comparison dashboard
- Deployment on Streamlit Cloud / AWS
- Add feature importance visualization

--- 

---

⭐ If you like this project, consider giving it a star!
