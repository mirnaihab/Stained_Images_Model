import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import requests
import cv2
import os

# Define the path to the model file in your GitLab repository
gitlab_raw_url = 'https://gitlab.com/mirnaihab/Stained_Images_Model/main/model.pkl'

# Retrieve the token from environment variables
gitlab_token = os.getenv('glpat-d7oawmWAbwtUc13K3Ebt')

def download_model(gitlab_raw_url, gitlab_token):
    headers = {'Private-Token': gitlab_token}
    response = requests.get(gitlab_raw_url, headers=headers)
    response.raise_for_status()  # Ensure we notice bad responses
    model_bytes = response.content
    print(f"Downloaded model size: {len(model_bytes)} bytes")  # Debugging info
    with open('downloaded_model.pkl', 'wb') as f:
        f.write(model_bytes)
    return 'downloaded_model.pkl'

try:
    # Load the model from GitLab
    model_path = download_model(gitlab_raw_url, gitlab_token)
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    print('Model loaded successfully')
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define the FastAPI application
app = FastAPI()

# Define the data model for prediction input
class PatientData(BaseModel):
    image_url: str

# Define the prediction endpoint
@app.post("/")
async def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

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
        return {"prediction": label_prediction.tolist()[0]}  # Convert to list for JSON serialization

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
# # Define the path to the model file on your local machine
# local_path = 'model.pkl'

# # Load the model from the specified path
# #model = joblib.load(local_path)

# # Verify the model is loaded
# print('Model loaded successfully')

# # Define the FastAPI application
# app = FastAPI()
 
# # Define the data model for prediction input
# class PatientData(BaseModel):
#     image_url: str

#     # Function to load the model
# def load_model():
#     # Download the model.pkl file from Git LFS
#     response = requests.get("https://github.com/mirnaihab/Stained_Images_Model/raw/main/model.pkl")
#     if response.status_code != 200:
#         raise HTTPException(status_code=500, detail="Failed to download the model file")

#     # Save the downloaded model.pkl file locally
#     with open(local_path, 'wb') as f:
#         f.write(response.content)

# # Load the model
# load_model()
# model = joblib.load(local_path)
# print('Model loaded successfully')


# # Define the prediction endpoint
# @app.post("/")
# async def predict(data: PatientData):
#     try:
#         # Print the image URL received
#         print(f"Received image URL: {data.image_url}")

#         # Download the image from the URL
#         response = requests.get(data.image_url)
#         if response.status_code != 200:
#             return {"error": "Failed to download the image from the provided URL"}

#         # Convert the downloaded image to a NumPy array
#         image = Image.open(BytesIO(response.content))
#         image_array = np.array(image)

#         # Preprocess the image
#         resized_image = cv2.resize(image_array, (300, 300))
#         flattened_image = resized_image.reshape(1, -1)
#         normalized_image = flattened_image.astype('float32') / 255.0

#         # Predict using the SVM model
#         label_prediction = model.predict(normalized_image)
#         return label_prediction.tolist()[0]  # Convert to list for JSON serialization

#     except Exception as e:
#         print(f"Error: {e}")
#         return {"error": str(e)}



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
