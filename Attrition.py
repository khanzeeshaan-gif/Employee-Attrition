import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and scaler
# These files must be in the same GitHub folder!
model = joblib.load('Emp_attrn_model.joblib')
le = joblib.load('Emp_attrn_LE.joblib')
scaler = joblib.load('Emp_attrn_scaler.joblib')

st.title("Employee Attrition Predictor")

# 2. Input fields
Age = st.number_input("Age",18,60,21)
JobLevel = st.number_input("JobLevel",1,5,2)
YearsAtCompany = st.number_input("YearsAtCompany",0,40,5)
MonthlyIncome = st.number_input("MonthlyIncome")
JobSatisfaction = st.number_input("JobSatisfaction",0,5,2)
WorkLifeBalance = st.number_input("WorkLifeBalance",1,5,4)
DistanceFromHome = st.number_input("DistanceFromHome")
PerformanceRating = st.number_input("PerformanceRating",0,5,4)
TrainingHoursLastYear = st.number_input("TrainingHoursLastYear")
Department = st.text_input("Engineering,Finance,HR,IT,Marketing,Operations,Sales")
OverTime = st.text_input("Enter Yes or No")
PromotionLast5Years = st.text_input("Enter Yes or No ONLY")

if st.button("Click here to get the Prediction"):
    # 3. Create feature list (Ensure order matches your training data!)
    x_cat=[Department,OverTime,PromotionLast5Years]
    x_num=[Age,JobLevel,YearsAtCompany,MonthlyIncome,JobSatisfaction,WorkLifeBalance,DistanceFromHome,PerformanceRating,TrainingHoursLastYear]
    
    # 4. Scale and Predict
    scaled_features=[]
    for i in x_num:
        scaled_features.append(scaler.transform(i))
    for i in x_cat:
        scaled_features.append(le.transform(i)
                           
    arr = np.array(scaled_features).reshape(1,12)
    prediction = model.predict(arr)
    
    if prediction[0] == 1:
        st.error("The model predicts a high risk of attrition.")
    else:
        st.success("The model predicts a low risk of attrition.")
