#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import cv2
import gdown
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile

app = FastAPI()

# Google Drive file ID for the Keras model
MODEL_FILE_ID = 'https://drive.google.com/uc?id1CkEfmk8hODpTpU5D4CWNJUDU7Wb3soGF'  # Replace this with the actual file ID from Google Drive
MODEL_PATH = 'building_dualchannel_model.keras'

# Download the Keras model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):  # Check if model exists locally
        print("Downloading the model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists locally.")

# Load model and class names on startup
download_model()  # Ensure the model is downloaded
model = load_model(MODEL_PATH)
class_names = np.load("dual_channel_class_names.npy")


def preprocess_image_dual_channel(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    img_no_green = cv2.bitwise_and(img, img, mask=mask_inv)

    gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    gray_resized = cv2.resize(gray, (224, 224))
    edges_resized = cv2.resize(edges, (224, 224))

    stacked = np.stack((gray_resized, edges_resized), axis=-1).astype('float32') / 255.0
    return np.expand_dims(stacked, axis=0)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name

        preprocessed = preprocess_image_dual_channel(temp_path)

        os.remove(temp_path)  # Clean up

        if preprocessed is None:
            return JSONResponse(content={"error": "Invalid image."}, status_code=400)

        prediction = model.predict(preprocessed)
        class_idx = np.argmax(prediction)
        class_name = class_names[class_idx]

        return {"predicted_class": class_name}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

