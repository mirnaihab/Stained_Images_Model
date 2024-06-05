import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import requests
import cv2

# Define the path to the model file on your local machine
local_path = 'model.pkl'

# Load the model from the specified path
model = joblib.load(local_path)

# Verify the model is loaded
print('Model loaded successfully')

# Define the FastAPI application
app = FastAPI()
 
# Define the data model for prediction input
class PatientData(BaseModel):
    image_url: str

# Define the prediction endpoint
@app.post("/")
async def predict(data: PatientData):
    try:
        # Print the image URL received
        print(f"Received image URL: {data.image_url}")

        # Download the image from the URL
        response = requests.get(data.image_url)
        if response.status_code != 200:
            return {"error": "Failed to download the image from the provided URL"}

        # Convert the downloaded image to a NumPy array
        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)

        # Preprocess the image
        resized_image = cv2.resize(image_array, (300, 300))
        flattened_image = resized_image.reshape(1, -1)
        normalized_image = flattened_image.astype('float32') / 255.0

        # Predict using the SVM model
        label_prediction = model.predict(normalized_image)
        return label_prediction.tolist()[0]  # Convert to list for JSON serialization

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}



# import joblib
# import numpy as np
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# import numpy as np

# import cv2

# #import cv2

# # Define the path to the model file on your local machine
# local_path = 'model.pkl'

# # Load the model from the specified path
# model = joblib.load(local_path)

# # Verify the model is loaded
# print('Model loaded successfully')

# # Define the FastAPI application
# app = FastAPI()
 
# # Define the data model for prediction input
# class PatientData(BaseModel):
#     X_path : str

# # Define the prediction endpoint
# @app.post("/")
# async def predict(data: PatientData):
#     try:
#         # Print the path received
#         print(f"Received X_path: {data.X_path}")

#         # Load and preprocess the image
#         X = cv2.imread(data.X_path)
#         if X is None:
#             return {"error": "Image not found at the specified path"}

#         X = cv2.resize(X, (300, 300))  # Ensure the image is resized to 300x300
#         X = X.reshape(1, -1)  # Flatten the image
#         X = X.astype('float32') / 255.0  # Normalize the pixel values

#         # Predict using the SVM model
#         label_prediction = model.predict(X)
#         return label_prediction.tolist()[0]  # Convert to list for JSON serialization

#     except Exception as e:
#         print(f"Error: {e}")
#         return {"error": str(e)}
