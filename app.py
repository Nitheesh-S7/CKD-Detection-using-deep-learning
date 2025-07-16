import streamlit as st
import numpy as np
import torch
import joblib
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.layer_1 = nn.Linear(9, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.layer_out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
    
# Load model and scaler
model = torch.load("model.pth", map_location=torch.device("cpu"),weights_only=False)
model.eval()
scaler = joblib.load("scaler.pkl")

st.title("CKD Prediction - AI Powered")

# Numeric inputs
hemo = st.number_input("Hemoglobin (hemo)", value=15.0, step=0.1)
sc = st.number_input("Serum Creatinine (sc)", value=1.2, step=0.1)
pvc = st.number_input("Packed Cell Volume (pvc)", value=44.0, step=1.0)
al = st.number_input("Albumin (al)", value=1.0, step=1.0)
rc = st.number_input("Red Blood Cell Count (rc)", value=4.5, step=0.1)
sg = st.number_input("Specific Gravity (sg)", value=1.020, step=0.001)
sod = st.number_input("Sodium (sod)", value=137.0, step=1.0)

# Boolean inputs (True/False)
htn = st.selectbox("Hypertension (htn)", options=[True, False])
dm = st.selectbox("Diabetes Mellitus (dm)", options=[True, False])

if st.button("Predict"):
    # Convert True/False to 1/0
   
    # Input order must match training
    input_data = [[hemo, sc, pvc, al, rc, htn, sg, dm, sod]]
    scaled_input = scaler.transform(input_data)
    tensor_input = torch.FloatTensor(scaled_input)

    with torch.no_grad():
        output = model(tensor_input)
        prob = torch.sigmoid(output).item()
        prediction = round(prob)

    st.subheader("Prediction Result")
    st.write("✅ **No CKD Detected**" if prediction == 0 else "🛑 **CKD Detected**")
    st.info(f"Model Confidence: {prob:.2%}") 
    
