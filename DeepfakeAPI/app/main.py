from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.model.model import (
    get_deepfake_model, predict_with_model
)

# Maximum file size (10MB limit)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Define class names
DEEPFAKE_CLASSES = ["Real Image", "Deepfake Detected"]


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_deepfake")
async def predict_deepfake(file: UploadFile = File(...)):
    """Predict Deepfake from X-ray image."""
    try:
        file_content = await file.read()
        logging.info(f"File size: {len(file_content)} bytes")

        model = get_deepfake_model()
        result = predict_with_model(model, file_content, DEEPFAKE_CLASSES)

        return {
            "filename": file.filename,
            "confidence": result["confidence"],
            "description": result["description"]
        }
    except HTTPException as e:
        logging.error(f"Deepfake prediction error: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected Deepfake prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
