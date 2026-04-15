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
JobLevel = st.number_input("Job Level",1,5,2)
YearsAtCompany = st.number_input("Years At Company",0,40,5)
MonthlyIncome = st.number_input("Monthly Income(rs)",min_value=1000)
JobSatisfaction = st.number_input("Job Satisfaction",0,5,2)
WorkLifeBalance = st.number_input("Work Life Balance",1,5,4)
DistanceFromHome = st.number_input("Distance From Home(in kms)",min_value=1)
PerformanceRating = st.number_input("Performance Rating",0,5,4)
TrainingHoursLastYear = st.number_input("Training Hours Last Year(hrs)")
Department = st.selectbox("Department",["Engineering","Finance","HR","IT","Marketing","Operations,Sales"])
OverTime = st.selectbox("Overtime", ["Yes","No"])
PromotionLast5Years = st.selectbox("Promotion Last 5Years",["Yes","No"])

if st.button("Click here to get the Prediction"):
    # 3. Create feature list (Ensure order matches your training data!)
    x_cat=[Department,OverTime,PromotionLast5Years]
    x_num=[Age,JobLevel,YearsAtCompany,MonthlyIncome,JobSatisfaction,WorkLifeBalance,DistanceFromHome,PerformanceRating,TrainingHoursLastYear]
    
    
    # 4. Scale and Predict
    scaled_features=[]
    x_num = scaler.transform(np.array(x_num).reshape(1,-1))[0]
    scaled_features.extend(x_num)
    for i in x_cat:
        if i in le.classes_:
            encoded= le.transform([i])[0]
        else:
            encoded = 0
        scaled_features.append(encoded)
                           
    arr = np.array(scaled_features).reshape(1,12)
    prediction = model.predict(arr)
    
    if prediction[0] == 1:
        st.error("The model predicts a high risk of attrition.")
    else:
        st.success("The model predicts a low risk of attrition.")
