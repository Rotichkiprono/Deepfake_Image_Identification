import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging
import pydicom
from fastapi import HTTPException
import os
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Preprocess image function
def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for model prediction."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.format not in ['JPEG', 'PNG']:
            raise ValueError("Unsupported image format. Only JPEG and PNG are supported.")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0).astype(np.float32)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

class ModelLoader:
    _deepfake_model = None

    @classmethod
    def get_deepfake_model(cls, model_path: str = '/app/app/model/deepfakemodel'):
        if cls._deepfake_model is None:
            try:
                logger.info(f"Loading Deepfake model from {model_path}")
                if not os.path.isdir(model_path):
                    raise FileNotFoundError(f"Model directory not found: {model_path}")
                cls._deepfake_model = tf.saved_model.load(model_path)
                logger.info("Deepfake model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Deepfake model: {e}")
                raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
        return cls._deepfake_model

# Prediction helper function
def predict_with_model(model, image_bytes: bytes, class_names: list) -> dict:
    """Make a prediction using the loaded TensorFlow SavedModel."""
    try:
        processed_image = preprocess_image(image_bytes)
        signature = model.signatures['serving_default']
        predictions = signature(tf.convert_to_tensor(processed_image))
        output_key = list(signature.structured_outputs.keys())[0]
        predictions = predictions[output_key].numpy()

        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])

        # Map confidence to medical descriptions
        if confidence < 0.30:
            description = "This is a real image."
        elif confidence < 0.75:
            description = "Possible traces of AI generation detected. Further analysis is advised."
        else:
            description = "Deepfake detected."

        return {
            "confidence": confidence,
            "description": description
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Expose the ModelLoader methods
get_deepfake_model = ModelLoader.get_deepfake_model

