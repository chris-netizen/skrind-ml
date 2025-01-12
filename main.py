import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
MODEL_PATH = 'feature_extraction_with_data_augmentation.keras'
try:
    model = keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# Define class labels (match the order of your dataset)
CLASS_LABELS = ['Invalid', 'Negative', 'Positive']

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess the uploaded image for prediction.
    """
    try:
        # Open the image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's input size (180x180)
        image = image.resize((180, 180))
        
        img_array = np.array(image)
        img_array = keras.applications.vgg16.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error in preprocessing image: {e}")
        raise ValueError(f"Error in preprocessing image: {e}")

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint to handle image prediction.
    """
    try:
        # Read image bytes
        image_bytes = await image.read()
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_LABELS[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Log detailed prediction information
        logger.info(f"Raw Predictions: {predictions[0]}")
        logger.info(f"Predicted Class: {predicted_class}")
        logger.info(f"Confidence: {confidence}")
        
        return JSONResponse({
            "prediction": predicted_class,
            "confidence": confidence,
            "all_probabilities": {label: float(prob) for label, prob in zip(CLASS_LABELS, predictions[0])},
            "status": "success"
        })
    
    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint to verify the API is running and the model is loaded.
    """
    return JSONResponse({
        "status": "healthy",
        "model_loaded": os.path.isfile(MODEL_PATH)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)