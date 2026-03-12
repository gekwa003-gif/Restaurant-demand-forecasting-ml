
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("🍽️ Restaurant Demand Forecasting")

# Generate synthetic data
dates = pd.date_range(start="2024-01-01", periods=180)
sales = np.random.randint(200, 500, size=180)

df = pd.DataFrame({
    "date": dates,
    "sales": sales
})

df["day_of_week"] = df["date"].dt.dayofweek

X = df[["day_of_week"]]
y = df["sales"]

model = RandomForestRegressor()
model.fit(X, y)

future_day = st.selectbox("Select Day of Week (0=Mon, 6=Sun)", list(range(7)))

prediction = model.predict([[future_day]])

st.write(f"### Predicted Sales: ${prediction[0]:.2f}")
