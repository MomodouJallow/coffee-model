import io

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from PIL import Image

# Define your class labels
class_labels = ["miner", "rust", "phome"]  

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input shape
print("Input shape:", input_details[0]['shape'])

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_size = (input_details[0]['shape'][2], input_details[0]['shape'][1])
        image.thumbnail(input_size, Image.LANCZOS)
        image = np.array(image.resize((input_size[0], input_size[1]), Image.LANCZOS), dtype=np.float32)
        image = image / 255.0
        input_data = np.zeros(input_details[0]['shape'], dtype=np.float32)
        input_data[0, :image.shape[0], :image.shape[1], :] = image

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data)
        predicted_class = class_labels[predicted_class_index]
        #Hello

        result = {
            "predicted_class": predicted_class,
            "confidence_score": float(output_data[0][predicted_class_index])
        }
    except Exception as e:
        result = {"Error": f"Failed to predict image: {str(e)}"}

    return jsonable_encoder(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)