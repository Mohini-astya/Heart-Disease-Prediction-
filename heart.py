import streamlit as st
import pickle
import numpy as np
from  sklearn.preprocessing import StandardScaler
import joblib

st.title("Heart Disease Data Input Form")



@st.cache_resource
def load_model():
    try:
        with open("D:/heart_disease/model1.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load vectorizer and model
model = load_model()

print("Model loaded:", model)

age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)


# Sex (Selectbox for Binary Input)
sex_map = {"Male": 1, "Female": 0}
sex = st.selectbox("Sex", options=list(sex_map.keys()))
sex_num = sex_map[sex]

# Chest Pain Type (Selectbox)
chest_pain_type_map = {
    "Typical Angina": 0, 
    "Atypical Angina": 1, 
    "Non-Anginal Pain": 2, 
    "Asymptomatic": 3
}
chest_pain_type = st.selectbox("Chest Pain Type", options=list(chest_pain_type_map.keys()))
chest_pain_type_num = chest_pain_type_map[chest_pain_type]

# Resting Blood Pressure (Numeric Input)
resting_blood_pressure = st.number_input(
    "Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1
)

# Cholesterol (Numeric Input)
cholestrol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)

# Fasting Blood Sugar (Selectbox for Binary Input)
fasting_blood_sugar_map = {"Yes": 1, "No": 0}
fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl", options=list(fasting_blood_sugar_map.keys())
)
fasting_blood_sugar_num = fasting_blood_sugar_map[fasting_blood_sugar]

# Resting Electrocardiographic Results (Selectbox)
resting_electrocardiographic_map = {
    "Normal": 0, 
    "ST-T Wave Abnormality": 1, 
    "Left Ventricular Hypertrophy": 2
}
resting_electrocardiographic = st.selectbox(
    "Resting Electrocardiographic Results", options=list(resting_electrocardiographic_map.keys())
)
resting_electrocardiographic_num = resting_electrocardiographic_map[resting_electrocardiographic]

# Maximum Heart Rate (Numeric Input)
maximum_heart_rate = st.number_input(
    "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1
)

# Exercise-induced Angina (Selectbox for Binary Input)
exercise_induced_angina_map = {"Yes": 1, "No": 0}
exercise_induced_angina = st.selectbox(
    "Exercise-induced Angina", options=list(exercise_induced_angina_map.keys())
)
exercise_induced_angina_num = exercise_induced_angina_map[exercise_induced_angina]

# ST Depression (Numeric Input)
st_depression = st.number_input(
    "ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1
)

# Slope of Peak Exercise ST Segment (Selectbox)
slope_peak_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope_peak = st.selectbox(
    "Slope of Peak Exercise ST Segment", options=list(slope_peak_map.keys())
)
slope_peak_num = slope_peak_map[slope_peak]

# Number of Major Vessels (Numeric Input)
number_of_major_vessels = st.number_input(
    "Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, step=1
)

# Thalassemia (Selectbox)
thalassemia_map = {
    "Normal": 0, 
    "Fixed Defect": 1, 
    "Reversible Defect": 2
}
thalassemia = st.selectbox(
    "Thalassemia", options=list(thalassemia_map.keys())
)
thalassemia_num = thalassemia_map[thalassemia]


# Combine Inputs into a NumPy Array
user_data = np.array([
    age,
    sex_num,
    chest_pain_type_num,
    resting_blood_pressure,
    cholestrol,
    fasting_blood_sugar_num,
    resting_electrocardiographic_num,
    maximum_heart_rate,
    exercise_induced_angina_num,
    st_depression,
    slope_peak_num,
    number_of_major_vessels,
    thalassemia_num
]).reshape(1, -1)


if st.button("predict"):

# Fit and transform the training data, transform the test data
    
    #scale_input = scaler.transform(user_data)  # Use transform, not fit_transform
    result = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]  # Probability of having heart disease

    #probability of having heart disease

    # Display the result
    if result == 1:
        st.write("### Prediction: High Risk of Heart Disease")
    else:
        st.write("### Prediction: Low Risk of Heart Disease")

    st.write(f"### Probability of Heart Disease: {probability * 100:.2f}%")    