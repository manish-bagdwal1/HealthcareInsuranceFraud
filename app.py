import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, precision_score, recall_score, confusion_matrix, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import streamlit as st

st.title(" Healthcare Fraud Web App")

image = Image.open('Fraud.jpg')

st.image(image,width = 450)
st.subheader('Intro to Healthcare Insurance fraud')
st.markdown("""
It is one of the biggest problems facing Medicare. In this type of fraud, false or **misleading information** is provided to a health insurance company.
Healthcare fraud is an **organized crime** which involves peers of **providers, physicians, beneficiaries acting together** to make fraud claims.
**Insurance companies** are the most **vulnerable** institutions impacted due to these bad practices.
""")

st.subheader('DataFrame')
df = pd.read_csv('train_final.csv')
st.write("Shape of the dataset:",df.shape)

st.dataframe(df)

def generate_data(x, models):
    
    res_x = []
    for model in models:
        res_x.append(model.predict(x))
    res_x = np.array(res_x).T
    
    return res_x

def final_fun_1(X):
    
    '''This function takes details about a healthcare provider as input and returns a prediction of the healthcare provider
       being a potential fraud. The details include: no. of inpatient claims(is_inpatient), no. of claims with group codes
       (is_groupcode), no. of claims with chronic illnesses like heartfailure, alzeimer, diabetes, etc., avg. deductible amt,
       avg. insurance amount reimbursed to the provider and avg. no. of days a patient was admitted under provider's care.'''
    
    # Loading Standard Scaler model to scale the data
    with open ('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Storing all provider ids separately
    provider_ids = X['Provider'].values

    X = X.drop('Provider',axis=1)
    X = X [['InscClaimAmtReimbursed',
 'PerProvider_mean_InscClaimAmtReimbursed',
 'PerOperatingPhysician_mean_InscClaimAmtReimbursed',
 'PerAttendingPhysician_mean_InscClaimAmtReimbursed',
 'Days_Admitted',
 'Hospitalization_Duration',
 'PerProvider_mean_Hospitalization_Duration',
 'ChronicCond_rheumatoidarthritis',
 'ChronicCond_ObstrPulmonary',
 'PerOtherPhysician_mean_Hospitalization_Duration',
 'PerAttendingPhysician_mean_DeductibleAmtPaid',
 'PerOtherPhysician_mean_InscClaimAmtReimbursed',
 'ChronicCond_stroke',
 'PerOperatingPhysician_mean_DeductibleAmtPaid',
 'ExtraClaimDays',
 'PerOperatingPhysician_mean_Hospitalization_Duration',
 'is_inpatient',
 'ChronicCond_Cancer',
 'ChronicCond_Alzheimer',
 'DeductibleAmtPaid']]
    
    # Scaling data
    X_scaled = scaler.transform(X)
    
    # Loading all base learners
    with open('BaseLearners.pkl', 'rb') as f:
                baseLearners = pickle.load(f)
    
    # Loading custom model
    custom_model = load('best_custom_model.joblib')

    # Predictions
    x_meta = generate_data(X_scaled, baseLearners)
    y_pred = custom_model.predict(x_meta)
    y_prob = custom_model.predict_proba(x_meta)
    
    all_predictions = pd.DataFrame(X)
    all_predictions['PotentialFraud'] = y_pred
    all_predictions.insert(0, "Provider", provider_ids)
    
    return all_predictions,y_pred

# defining data and target variable
Y = df['PotentialFraud'].values

X = df.drop('PotentialFraud',axis=1)

st.sidebar.subheader("Select and click predict button")

row_no = st.sidebar.slider('Which datapoint you want select for prediction?', 0, 5409)

result=st.sidebar.button('Predict')

if result : 
    st.sidebar.write(":smile:")
    data_point = X[row_no : row_no+1]

    predict_df,output = final_fun_1(data_point)

    st.subheader('Model Prediction')
    st.write('For the selected datpoint model prediction is :',output[0])

    st.subheader("Output Dataframe")
    st.dataframe(predict_df)

