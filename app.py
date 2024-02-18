import io
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# Define your class labels
class_labels = ['miner', 'nodisease', 'phoma', 'rust']

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: bytes = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(io.BytesIO(file)).convert("RGB")
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

        result = {
            "predicted_class": predicted_class,
            "confidence_score": float(output_data[0][predicted_class_index])
        }
    except Exception as e:
        result = {"Error": f"Failed to predict image: {str(e)}"}

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
