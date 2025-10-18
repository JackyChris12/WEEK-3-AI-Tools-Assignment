from PIL import Image
import numpy as np


def pil_to_model_array(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to the (1,28,28,1) array expected by the model."""
    img = img.convert("L")
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr
