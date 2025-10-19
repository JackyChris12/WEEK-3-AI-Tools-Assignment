import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


@st.cache_data
def load_data(path: str = "archive/Iris.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["Id"]) if "Id" in df.columns else df
    df.dropna(inplace=True)
    return df


@st.cache_data
def train_model(df: pd.DataFrame):
    le = LabelEncoder()
    df_copy = df.copy()
    df_copy['Species'] = le.fit_transform(df_copy['Species'])

    X = df_copy.drop(columns=['Species'])
    y = df_copy['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    return {
        'model': clf,
        'label_encoder': le,
        'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall},
        'feature_stats': X.describe()
    }


def main():
    st.set_page_config(page_title="Iris Species Predictor", layout="centered")
    st.title("Iris Species Classification (Decision Tree)")

    df = load_data()
    artifacts = train_model(df)
    clf = artifacts['model']
    le = artifacts['label_encoder']
    metrics = artifacts['metrics']
    stats = artifacts['feature_stats']

    st.sidebar.header("Input features")
    def slider_for(col: str):
        col_stats = stats[col]
        return st.sidebar.slider(
            label=col,
            min_value=float(col_stats['min']),
            max_value=float(col_stats['max']),
            value=float(col_stats['mean']),
        )

    sepal_length = slider_for('SepalLengthCm')
    sepal_width = slider_for('SepalWidthCm')
    petal_length = slider_for('PetalLengthCm')
    petal_width = slider_for('PetalWidthCm')

    if st.sidebar.button('Predict'):
        X_new = [[sepal_length, sepal_width, petal_length, petal_width]]
        pred = clf.predict(X_new)[0]
        species = le.inverse_transform([pred])[0]
        st.success(f"Predicted species: {species}")

    st.markdown("---")
    st.subheader("Model evaluation on test split")
    st.write(f"Accuracy: {metrics['accuracy']:.4f}")
    st.write(f"Precision: {metrics['precision']:.4f}")
    st.write(f"Recall: {metrics['recall']:.4f}")

    st.subheader("Dataset sample")
    st.dataframe(df.head())


if __name__ == '__main__':
    main()
