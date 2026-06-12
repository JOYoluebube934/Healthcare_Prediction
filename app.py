import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="Healthcare Prediction System",
    page_icon="🏥",
    layout="wide"
)

# 2. Cache the model loading so it doesn't reload on every button click
@st.cache_resource
def load_model():
    return jb.load("healthcare_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'healthcare_model.pkl' not found. Please ensure it is in your repository.")
    st.stop()

# 3. App Title & Intro
st.title("🏥 Healthcare Disease Prediction System")
st.markdown("Fill out the patient details below and click **Generate Prediction**.")
st.write("---")

# 4. Organizing Inputs into Columns to reduce scrolling
st.markdown("### 📋 Patient Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    systolic_bp = st.number_input("Systolic BP", 80, 250, 120)
    diastolic_bp = st.number_input("Diastolic BP", 40, 150, 80)
    blood_glucose = st.number_input("Blood Glucose (mg/dL)", 50, 500, 100)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.5)
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75)

with col2:
    temperature = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0)
    diagnosis = st.selectbox("Diagnosis", ["Diabetes", "Hypertension", "Asthma", "Heart Disease"])
    department = st.selectbox("Department", ["Cardiology", "Neurology", "General Medicine", "Orthopedics"])
    length_of_stay = st.number_input("Length of Stay (Days)", 0, 100, 1)
    num_medications = st.number_input("Number of Medications", 0, 50, 2)
    num_comorbidities = st.number_input("Number of Comorbidities", 0, 20, 0)
    prev_admissions = st.number_input("Previous Admissions", 0, 50, 0)
    blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

with col3:
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    alcohol_use = st.selectbox("Alcohol Use", ["Never", "Occasional", "Regular"])
    exercise_frequency = st.selectbox("Exercise Frequency", ["Never", "Rarely", "Weekly", "Daily"])
    education_level = st.selectbox("Education Level", ["Primary", "Secondary", "Tertiary"])
    urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
    insurance_type = st.selectbox("Insurance Type", ["Private", "Public", "None"])
    admission_type = st.selectbox("Admission Type", ["Emergency", "Routine", "Referral"])

# 5. Mapping Dictionaries
mappings = {
    "gender": {"Male": 1, "Female": 0},
    "diagnosis": {"Diabetes": 0, "Hypertension": 1, "Asthma": 2, "Heart Disease": 3},
    "department": {"Cardiology": 0, "Neurology": 1, "General Medicine": 2, "Orthopedics": 3},
    "smoker": {"Yes": 1, "No": 0},
    "alcohol": {"Never": 0, "Occasional": 1, "Regular": 2},
    "exercise": {"Never": 0, "Rarely": 1, "Weekly": 2, "Daily": 3},
    "education": {"Primary": 0, "Secondary": 1, "Tertiary": 2},
    "urban_rural": {"Urban": 1, "Rural": 0},
    "insurance": {"Private": 0, "Public": 1, "None": 2},
    "admission": {"Emergency": 0, "Routine": 1, "Referral": 2},
    "blood": {"A+": 0, "A-": 1, "B+": 2, "B-": 3, "AB+": 4, "AB-": 5, "O+": 6, "O-": 7}
}

# 6. Transform the data for the model
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [mappings["gender"][gender]],
    "BMI": [bmi],
    "Systolic_BP": [systolic_bp],
    "Diastolic_BP": [diastolic_bp],
    "Blood_Glucose_mgdL": [blood_glucose],
    "Hemoglobin_gdL": [hemoglobin],
    "Heart_Rate_bpm": [heart_rate],
    "Temperature_C": [temperature],
    "Diagnosis": [mappings["diagnosis"][diagnosis]],
    "Department": [mappings["department"][department]],
    "Length_of_Stay_Days": [length_of_stay],
    "Num_Medications": [num_medications],
    "Num_Comorbidities": [num_comorbidities],
    "Prev_Admissions": [prev_admissions],
    "Smoker": [mappings["smoker"][smoker]],
    "Alcohol_Use": [mappings["alcohol"][alcohol_use]],
    "Exercise_Frequency": [mappings["exercise"][exercise_frequency]],
    "Education_Level": [mappings["education"][education_level]],
    "Urban_Rural": [mappings["urban_rural"][urban_rural]],
    "Insurance_Type": [mappings["insurance"][insurance_type]],
    "Admission_Type": [mappings["admission"][admission_type]],
    "Blood_Type": [mappings["blood"][blood_type]]
})

st.write("---")

# 7. Action Button prevents auto-running on layout render
if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
    with st.spinner("Processing medical data..."):
        prediction = model.predict(input_data)
        
        st.markdown("### 📊 Prediction Result")
        
        # Conditional Statement for Admission Status
        if prediction[0] == 0:
            st.success("🟢 **Status: Not Admitted**\nThe patient does not require hospital admission based on the current metrics.")
        elif prediction[0] == 1:
            st.error("🚨 **Status: Admitted**\nThe patient's health profile indicates they require hospital admission.")
        else:
            # Fallback in case your model outputs unexpected values
            st.warning(f"⚠️ **Status Unknown:** Model returned an unexpected value: {prediction[0]}")
