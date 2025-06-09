import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 


#Load and prepare the data
df = pd.read_csv('data/insurance.csv')
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Streamlit UI
st.title("Insurance Premium Predictor")

st.write("Fill in the details to estimate insurance cost.")

age = st.slider("Age", 18, 65, 30)
bmi = st.slider("BMI", 10.0, 45.0, 25.0)
children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
sex = st.radio("Sex", ['Male', 'Female'])
smoker = st.radio("Smoker", ['Yes', 'No'])
region = st.selectbox("Region", ['Northeast', 'Northwest', 'Southeast', 'Southwest'])

#Prepare input data
input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_male': [1 if sex == 'Male' else 0],
    'smoker_yes': [1 if smoker == 'Yes' else 0],
    'region_northwest': [1 if region == 'Northwest' else 0],
    'region_southeast': [1 if region == "Southeast" else 0],
    'region_southwest': [1 if region == "Southwest" else 0]
})

#Predict 
prediction = model.predict(input_data)[0]
st.subheader(f"Estimated Annual Insurance Cost: ${prediction:,.2f}")







