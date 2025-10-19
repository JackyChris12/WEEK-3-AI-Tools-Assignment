import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# App title
st.title("Iris Species Predictor")
st.write("Enter measurements for an iris flower and get a predicted species.")

# Load model and encoder
MODEL_PATH = Path(__file__).parent / "models" / "decision_tree.joblib"
ENCODER_PATH = Path(__file__).parent / "models" / "label_encoder.joblib"

@st.cache_resource
def load_artifacts():
	clf = joblib.load(MODEL_PATH)
	le = joblib.load(ENCODER_PATH)
	return clf, le

clf, le = load_artifacts()

# Sidebar inputs
st.sidebar.header("Input features")
sepal_length = st.sidebar.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1, format="%.2f")
sepal_width = st.sidebar.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1, format="%.2f")
petal_length = st.sidebar.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1, format="%.2f")
petal_width = st.sidebar.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1, format="%.2f")

input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
						columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]) 

st.subheader("Input preview")
st.write(input_df)

if st.button("Predict species"):
	pred_label = clf.predict(input_df)[0]
	pred_name = le.inverse_transform([pred_label])[0]
	st.success(f"Predicted species: {pred_name} (label {pred_label})")

st.markdown("---")
st.write("Model and encoder loaded from `models/` folder. If you need to retrain or replace them, update the `models` artifacts.")

