# salary_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv(r'C:\Users\USER\OneDrive\Documents\ardent\ardent\salary_Predictor.csv')

# Train model
X = df[['YearsExperience']]
y = df['Salary']
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("💼 Salary Prediction App")
st.write("Estimate salary based on years of experience using a linear regression model.")

# Input
exp = st.slider("Years of Experience", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

# Prediction
predicted_salary = model.predict(np.array([[exp]]))[0]
st.subheader("📈 Predicted Salary")
st.write(f"Estimated Salary: **${predicted_salary:,.2f}**")

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label="Actual Data")
ax.plot(X, model.predict(X), color='red', label="Regression Line")
ax.scatter(exp, predicted_salary, color='green', label="Your Input", zorder=5)
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary vs Experience")
ax.legend()
st.pyplot(fig)
