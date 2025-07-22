import streamlit as st
import pandas as pd
import numpy as np
import os
import dill
import time
import plotly.express as px
from src.utils import load_model

# Load Model
@st.cache_resource
def load_model_from_path():
    model_path = os.path.join("artifacts", "model.pkl")
    if os.path.exists(model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("Model not found. Please ensure artifacts/model.pkl exists.")
        return None

# Load MAE Metric
def load_mae():
    mae_path = os.path.join("artifacts", "mae.txt")
    if os.path.exists(mae_path):
        with open(mae_path, "r") as file:
            content = file.read().strip()
            try:
                return float(content.split(":")[-1].strip())
            except ValueError:
                st.warning("Invalid MAE format.")
                return None
    return None

# Page Config
st.set_page_config(
    page_title="Employee Salary Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.markdown("## 📍 Navigation")
page = st.sidebar.radio("Choose Page", ["🏠 Home", "📊 Predict Salary", "📈 Model Performance", "ℹ️ About Project"])

# Load model once
model = load_model_from_path()

# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("👨‍💼 Employee Salary Prediction Dashboard")
    st.markdown("""
    Welcome to the Employee Salary Prediction system!  
    This project predicts whether an individual earns **more than $50K/year** based on demographics and work features.

    ### 🔧 Technologies Used:
    - Python, Pandas, Scikit-learn
    - Streamlit for UI
    - Plotly for charts
    - Modular pipeline & serialization

    **Model Output:** Binary Classification: `>50K` or `<=50K`
    """)
    st.image("assets/dashboard_banner.jpg", use_container_width=True)


# --- PREDICTION PAGE ---
elif page == "📊 Predict Salary":
    st.title("📊 Predict Employee Salary")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 70, 30)
            education_num = st.slider("Education Number", 1, 16, 10)
            capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        with col2:
            hours_per_week = st.slider("Hours per Week", 1, 100, 40)
            workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
                                                   'Federal-gov', 'Local-gov', 'State-gov',
                                                   'Without-pay', 'Never-worked'])
            education = st.selectbox("Education", ['Bachelors', 'Some-college', '11th', 'HS-grad',
                                                   'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th'])
        with col3:
            marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                             'Separated', 'Widowed', 'Married-spouse-absent'])
            occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                                     'Exec-managerial', 'Prof-specialty'])
            gender = st.radio("Gender", ['Male', 'Female'])

        submitted = st.form_submit_button("🎯 Predict Now")

    if submitted:
        if model:
            input_df = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'education': [education],
                'education-num': [education_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'gender': [gender],
                'capital-gain': [capital_gain],
                'hours-per-week': [hours_per_week]
            })
            with st.spinner("Analyzing data and predicting..."):
                time.sleep(1.5)
                prediction = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][prediction]

                st.success(f"🎉 Predicted Salary Class: **{'>50K' if prediction else '<=50K'}**")
                st.metric("Model Confidence", f"{prob * 100:.2f}%")

                st.subheader("🔍 Input Summary")
                st.dataframe(input_df.T, use_container_width=True)
        else:
            st.error("Model not loaded.")

# --- PERFORMANCE PAGE ---
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")

    mae = load_mae()
    if mae:
        st.metric("Mean Absolute Error", round(mae, 3))
    else:
        st.warning("MAE metric not available.")

    st.subheader("🎯 Feature Importance (Sample)")
    importance_df = pd.DataFrame({
        'Feature': ['age', 'education-num', 'hours-per-week', 'capital-gain'],
        'Importance': [0.25, 0.22, 0.19, 0.34]
    })
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importances')
    st.plotly_chart(fig)

# --- ABOUT PAGE ---
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")
    st.markdown("""
    This interactive ML dashboard predicts employee salary classification based on attributes such as age, education, workclass, and hours per week.

    ### 💼 Ideal For:
    - Data Science Portfolio
    - Job Applications
    - Resume Projects

    ### 📁 Project Structure:
    ```
    Employee_Salary_Prediction/
    ├── artifacts/
    │   ├── model.pkl
    │   └── mae.txt
    ├── assets/
    │   └── dashboard_banner.jpg
    ├── src/
    │   └── utils.py
    └── streamlit_app.py
    ```

    ### 🚀 Next Steps:
    - Train the model using `train.py` script (you can ask me to write one!)
    - Add live feedback collection
    - Deploy to Streamlit Cloud or Hugging Face

    > Built with ❤️ using Python and Streamlit
    """)