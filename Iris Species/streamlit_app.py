import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


@st.cache_data
def load_artifacts(model_dir='models'):
    model_file = os.path.join(model_dir, 'decision_tree.joblib')
    encoder_file = os.path.join(model_dir, 'label_encoder.joblib')
    if not os.path.exists(model_file) or not os.path.exists(encoder_file):
        raise FileNotFoundError('Model or encoder not found. Run `iris_classification.py` to train and save them.')
    clf = joblib.load(model_file)
    le = joblib.load(encoder_file)
    return clf, le


def main():
    st.title('Iris Species Classifier')
    st.write('Enter the measurements of the iris flower and click Predict')

    sepal_length = st.slider('Sepal Length (cm)', 0.0, 10.0, 5.8)
    sepal_width = st.slider('Sepal Width (cm)', 0.0, 10.0, 3.0)
    petal_length = st.slider('Petal Length (cm)', 0.0, 10.0, 4.35)
    petal_width = st.slider('Petal Width (cm)', 0.0, 10.0, 1.3)

    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    try:
        clf, le = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    if st.button('Predict'):
        pred = clf.predict(input_df)
        pred_proba = clf.predict_proba(input_df)
        species = le.inverse_transform(pred)[0]
        proba = np.max(pred_proba)
        st.success(f'Predicted species: {species}')
        st.info(f'Prediction confidence: {proba:.2f}')


if __name__ == '__main__':
    main()
