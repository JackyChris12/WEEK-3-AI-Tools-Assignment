# MNIST CNN Streamlit App

This small Streamlit app loads the trained `mnist_cnn_model.h5` and provides a simple UI to upload an image of a handwritten digit and get a prediction.

Prerequisites
- Python 3.8+ recommended
- Windows PowerShell (instructions below) or any shell

Setup (PowerShell)

```powershell
# 1. Create a virtual environment (recommended)
python -m venv .venv
# 2. Activate the venv
.\.venv\Scripts\Activate.ps1
# 3. Upgrade pip
python -m pip install --upgrade pip
# 4. Install requirements
pip install -r requirements.txt
```

Run the app

```powershell
streamlit run app.py
```

Notes
- Ensure `mnist_cnn_model.h5` is in the same folder as `app.py`.
- The app currently supports image upload (PNG/JPG). Drawing canvas support can be added if you install `streamlit-drawable-canvas` and we update `app.py` accordingly.

Troubleshooting
- If TensorFlow install is slow or fails on Windows, consider installing a prebuilt wheel or using Anaconda.
- If you get GPU driver errors, install CPU-only TensorFlow via `pip install tensorflow-cpu` or follow TensorFlow GPU setup docs.

If you want, I can add a drawing canvas and sample images next. 
