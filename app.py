import io

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from PIL import Image

# Define your class labels
class_labels = ["miner", "rust", "phome"]

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize image to the input size required by the model
        input_size = (224, 224)  # Example input size, adjust according to your model
        image = image.resize(input_size, Image.LANCZOS)

        # Convert image to numpy array and normalize
        image = np.array(image, dtype=np.float32) / 255.0

        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path="models/converted_model.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], [image])

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Post-process output
        predicted_class_index = np.argmax(output_data)
        predicted_class = class_labels[predicted_class_index]

        result = {
            "predicted_class": predicted_class,
            "confidence_score": float(output_data[0][predicted_class_index])
        }
    except Exception as e:
        result = {"error": f"Failed to predict image: {str(e)}"}

    return jsonable_encoder(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
