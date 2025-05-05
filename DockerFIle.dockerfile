# Use official Python 3.9 image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies:
# libgl1 - Required for OpenCV
# wget - For downloading your model
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL other files (except those in .dockerignore)
COPY . .

# Download your Keras model from Google Drive
# Replace MODEL_FILE_ID with your actual ID
RUN wget --no-check-certificate \
    "https://drive.google.com/uc?export=download&id=1CkEfmk8hODpTpU5D4CWNJUDU7Wb3soGF" \
    -O "building_dualchannel_model.keras"

# Verify the model downloaded correctly
RUN python -c "from tensorflow.keras.models import load_model; load_model('building_dualchannel_model.keras')" || \
    { echo "❌ Model failed to load!"; exit 1; }

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]