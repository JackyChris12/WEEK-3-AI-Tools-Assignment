import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

st.title("MNIST Handwritten Digit Classifier")

st.write("Upload a grayscale image of a digit (28x28) or draw one below. The model expects 28x28 pixels, single channel, normalized to [0,1].")

# Sidebar
st.sidebar.header("Options")
use_upload = st.sidebar.checkbox("Upload image", value=True)

@st.cache_resource
def load_model(path="mnist_cnn_model.h5"):
    model = tf.keras.models.load_model(path)
    return model

# Load model
try:
    model = load_model("mnist_cnn_model.h5")
except Exception as e:
    st.error(f"Failed to load model: {e}\nMake sure 'mnist_cnn_model.h5' exists in the app folder.")
    st.stop()


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert PIL image to model-ready numpy array of shape (1,28,28,1)"""
    # Convert to grayscale
    img = img.convert("L")
    # Resize to 28x28
    img = img.resize((28, 28))
    # Invert colors if background is white and digit is dark
    # We assume digit is darker than background; ensure background is 0 and digit is 1
    arr = np.array(img).astype("float32")
    # Normalize to [0,1]
    arr = arr / 255.0
    # If background is white (close to 1), invert so that digit is high values
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    arr = np.expand_dims(arr, axis=0)   # (1,28,28,1)
    return arr


if use_upload:
    uploaded = st.file_uploader("Upload an image (PNG/JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Uploaded image", use_column_width=False)
        data = preprocess_image(image)
        preds = model.predict(data)
        pred_label = np.argmax(preds, axis=1)[0]
        confidence = preds[0, pred_label]
        st.success(f"Predicted: {pred_label} (confidence: {confidence:.2f})")
else:
    st.write("Drawing canvas is not available in minimal dependency mode. Please upload an image.")

st.markdown("---")
st.write("Tips:\n- Use a square image with a dark digit on a light background or vice versa.\n- If your digit looks inverted (white on black), the app will try to detect and invert automatically.")
