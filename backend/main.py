
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import os

app = FastAPI(title="Skin Cancer Prediction API")

# Get the correct model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "model",
        "skin_cancer_model.h5"
    )
)

# Disease classes
CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# Load trained model
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = load_model(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Skin Cancer API is running successfully!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Check uploaded file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a valid image."
            )

        # Read uploaded image
        image_data = await file.read()

        # Open and convert image to RGB
        image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

        # Resize image
        image = image.resize((224, 224))

        # Convert image to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Make prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        # Get predicted class and confidence
        predicted_index = int(
            np.argmax(prediction[0])
        )

        confidence = float(
            np.max(prediction[0]) * 100
        )

        predicted_class = CLASS_NAMES[predicted_index]

        return {
            "disease": predicted_class,
            "confidence": round(confidence, 2),
            "message": (
                "AI prediction only. This is not a medical diagnosis. "
                "Please consult a qualified dermatologist."
            )
        }

    except HTTPException:
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check the server logs."
        )

    finally:
        await file.close()
