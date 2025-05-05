#!/usr/bin/env python
import os
import numpy as np
import cv2
import gdown
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
from typing import Optional

app = FastAPI(title="Building Classifier API")

# Configuration (now using environment variables)
MODEL_FILE_ID = os.getenv('MODEL_FILE_ID', '1CkEfmk8hODpTpU5D4CWNJUDU7Wb3soGF')
MODEL_PATH = os.getenv('MODEL_PATH', 'building_dualchannel_model.keras')
CLASS_NAMES_PATH = os.getenv('CLASS_NAMES_PATH', 'dual_channel_class_names.npy')

# Global variables for loaded model
model = None
class_names = None

@app.on_event("startup")
async def load_model_and_classes():
    """Load model and class names during startup"""
    global model, class_names
    
    try:
        # Download model if not exists (with progress)
        if not os.path.exists(MODEL_PATH):
            print("⬇️ Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        
        # Load model and classes
        model = load_model(MODEL_PATH)
        class_names = np.load(CLASS_NAMES_PATH)
        print("✅ Model and classes loaded successfully")
        
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Model not loaded"
        )
    return {"status": "healthy", "model_loaded": model is not None}

def preprocess_image_dual_channel(image_path: str) -> Optional[np.ndarray]:
    """Preprocess image for dual-channel model"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Channel 1: Grayscale without green areas
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        img_no_green = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        gray = cv2.cvtColor(img_no_green, cv2.COLOR_BGR2GRAY)

        # Channel 2: Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), iterations=1)

        # Resize and normalize
        gray_resized = cv2.resize(gray, (224, 224))
        edges_resized = cv2.resize(edges, (224, 224))
        stacked = np.stack((gray_resized, edges_resized), axis=-1).astype('float32') / 255.0

        return np.expand_dims(stacked, axis=0)
    except Exception:
        return None

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle image prediction"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name

        # Preprocess and predict
        preprocessed = preprocess_image_dual_channel(temp_path)
        os.remove(temp_path)  # Cleanup

        if preprocessed is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image or preprocessing failed"
            )

        prediction = model.predict(preprocessed)
        class_idx = np.argmax(prediction)
        
        return {
            "predicted_class": str(class_names[class_idx]),
            "confidence": float(np.max(prediction)),
            "all_predictions": prediction.tolist()[0]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
