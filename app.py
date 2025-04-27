import streamlit as st
import joblib
import numpy as np

# Load models
parkinsons_model = joblib.load('models/parkinsons_model.pkl')
kidney_model = joblib.load('models/kidney_model.pkl')
liver_model = joblib.load('models/liver_model.pkl')

# Prediction function
def predict_disease(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Set Streamlit page config
st.set_page_config(page_title="Disease Predictor", page_icon="🔬", layout="wide")
st.title("👩🏻‍⚕️ Disease Prediction Portal")

# Apply Custom Light Theme CSS
st.markdown("""
    <style>
        body {
            background-color: #fafafa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stApp {
            background-color: #ffffff;
        }
        h1 {
            color: #b30000;
            font-weight: bold;
            padding-bottom: 10px;
        }
        h3 {
            background-color: #ffcccc;
            color: #660000;
            padding: 18px;
            border-radius: 10px;
            font-size: 24px;
            font-weight: 600;
            width: fit-content;
            margin: 20px 0 20px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton button {
            background-color: #b30000;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #990000;
            color: #ffffff;
        }
        .stSelectbox, .stTextInput {
            margin-bottom: 20px;
        }
        .stTextInput input, .stSelectbox select {
            background-color: #f7f7f7;
            border: 1px solid #cccccc;
            padding: 10px;
            border-radius: 8px;
            font-size: 16px;
            color: #333333;
        }
        .sidebar .sidebar-content {
            background-color: #ffe6e6;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🩺 Choose Disease")
disease_option = st.sidebar.selectbox(
    "Select Disease to Predict",
    ["Parkinson's Disease", "Kidney Disease", "Liver Disease"]
)

# Disease Specific Inputs
if disease_option == "Parkinson's Disease":
    st.markdown('<h3>🧠 Parkinson\'s Disease Prediction </h3>', unsafe_allow_html=True)
    
    input_data = [
        float(st.text_input('PPE', value='0.0')),
        float(st.text_input('MDVP:Fo(Hz)', value='0.0')),
        float(st.text_input('spread1', value='0.0')),
        float(st.text_input('MDVP:Flo(Hz)', value='0.0')),
        float(st.text_input('jitter:DDP', value='0.0')),
        float(st.text_input('MDVP:Fhi(Hz)', value='0.0')),
        float(st.text_input('spread2', value='0.0'))
    ]
    
    if st.button("Predict Parkinson's Disease"):
        prediction = predict_disease(parkinsons_model, input_data)
        if prediction == 1:
            st.success("✅ Prediction: Parkinson's disease detected.")
        else:
            st.success(" ❌ Prediction: No Parkinson's disease detected.")

elif disease_option == "Kidney Disease":
    st.markdown('<h3>🫘 Kidney Disease Prediction </h3>', unsafe_allow_html=True)
    
    input_data = [
        float(st.text_input('hemo', value='0.0')),
        float(st.text_input('sc', value='0.0')),
        float(st.text_input('sg', value='0.0')),
        float(st.text_input('pcv', value='0.0')),
        float(st.text_input('al', value='0.0')),
        st.selectbox('dm (Diabetes Mellitus)', ['Yes', 'No'])
    ]
    
    if st.button("Predict Kidney Disease"):
        input_data[5] = 1 if input_data[5] == 'Yes' else 0
        prediction = predict_disease(kidney_model, input_data)
        if prediction == 1:
            st.success("✅ Prediction: Kidney disease detected.")
        else:
            st.success("❌ Prediction: No Kidney disease detected.")

elif disease_option == "Liver Disease":
    st.markdown('<h3>🍠 Liver Disease Prediction </h3>', unsafe_allow_html=True)
    
    input_data = [
        float(st.text_input('Alkaline Phosphotase', value='0.0')),
        float(st.text_input('Aspartate Aminotransferase', value='0.0')),
        float(st.text_input('Alamine Aminotransferase', value='0.0')),
        float(st.text_input('Age', value='0.0')),
        float(st.text_input('Total Bilirubin', value='0.0')),
        float(st.text_input('Total Proteins', value='0.0')),
        float(st.text_input('Albumin', value='0.0')),
        float(st.text_input('Direct Bilirubin', value='0.0')),
        float(st.text_input('Albumin and Globulin Ratio', value='0.0'))
    ]
    
    if st.button("Predict Liver Disease"):
        prediction = predict_disease(liver_model, input_data)
        if prediction == 1:
            st.success("✅ Prediction: Liver disease detected.")
        else:
            st.success("❌ Prediction: No Liver disease detected.")
