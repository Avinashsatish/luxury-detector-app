from flask import Flask, request, render_template, jsonify
from PIL import Image
import cv2
import numpy as np
import os
import requests
import logging

app = Flask(__name__)

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

# === CONFIG ===
CUSTOM_VISION_PREDICTION_KEY = "7j4giuRUiS8V4ycv52wfMoCkPYyBZXUrvTXaW00P2mLrRghP7lpRJQQJ99BFACYeBjFXJ3w3AAAIACOGg5aY"
CUSTOM_VISION_ENDPOINT = "https://project4823-prediction.cognitiveservices.azure.com/"
CUSTOM_VISION_PROJECT_ID = "961c6cf3-471c-423b-9726-2b9bea7f75f3"
CUSTOM_VISION_PUBLISHED_NAME = "Iteration1"

VISION_API_KEY = "8fhbv1SOfwdTiUvt2zZgfZfVIxGOIcraOZvyo1NUnrSaCOnMDr5wJQQJ99BFACYeBjFXJ3w3AAAFACOGTZPK"
VISION_API_ENDPOINT = "https://project4823-checkimage.cognitiveservices.azure.com/"

@app.route('/')
def home():
    return render_template('index.html')

def is_blurry(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        os.makedirs('uploads', exist_ok=True)
        save_path = os.path.join('uploads', image_file.filename)
        image_file.save(save_path)

        # Get basic image info
        img = Image.open(save_path)
        width, height = img.size
        format_type = img.format
        blurry = is_blurry(save_path)

        # === Call Azure Computer Vision for description ===
        description = ""
        vision_url = f"{VISION_API_ENDPOINT}/vision/v3.2/analyze?visualFeatures=Description"
        vision_headers = {
            "Ocp-Apim-Subscription-Key": VISION_API_KEY,
            "Content-Type": "application/octet-stream"
        }
        
        try:
            with open(save_path, 'rb') as img_data:
                vision_response = requests.post(vision_url, headers=vision_headers, data=img_data)
            
            if vision_response.status_code == 200:
                captions = vision_response.json().get("description", {}).get("captions", [])
                if captions:
                    description = captions[0]["text"]
            else:
                app.logger.error(f"Vision API Error: {vision_response.status_code} - {vision_response.text}")
        except Exception as e:
            app.logger.error(f"Vision API Exception: {str(e)}")

        # === Call Custom Vision API for prediction ===
        prediction_tag = "Unknown"
        prediction_confidence = 0.0
        
        # Fixed Custom Vision URL and headers
        prediction_url = f"{CUSTOM_VISION_ENDPOINT}/customvision/v3.0/Prediction/{CUSTOM_VISION_PROJECT_ID}/classify/iterations/{CUSTOM_VISION_PUBLISHED_NAME}/image"
        cv_headers = {
            "Prediction-Key": CUSTOM_VISION_PREDICTION_KEY,
            "Content-Type": "application/octet-stream"
        }

        try:
            with open(save_path, 'rb') as image_data:
                pred_response = requests.post(prediction_url, headers=cv_headers, data=image_data)
            
            app.logger.debug(f"Custom Vision Response Status: {pred_response.status_code}")
            app.logger.debug(f"Custom Vision Response: {pred_response.text}")

            if pred_response.status_code == 200:
                response_data = pred_response.json()
                predictions = response_data.get("predictions", [])
                
                if predictions:
                    best = max(predictions, key=lambda x: x["probability"])
                    prediction_tag = best["tagName"]
                    prediction_confidence = best["probability"]
                else:
                    prediction_tag = "No predictions available"
                    prediction_confidence = 0.0
            else:
                prediction_tag = f"API Error: {pred_response.status_code}"
                prediction_confidence = 0.0
                app.logger.error(f"Custom Vision API Error: {pred_response.status_code} - {pred_response.text}")
                
        except Exception as e:
            prediction_tag = f"Exception: {str(e)}"
            prediction_confidence = 0.0
            app.logger.error(f"Custom Vision Exception: {str(e)}")

        result = {
            'format': format_type,
            'width': width,
            'height': height,
            'blurry': 'Yes' if blurry else 'No',
            'caption': description,
            'prediction': prediction_tag,
            'prediction_confidence': round(prediction_confidence * 100, 2)
        }

        # Clean up uploaded file
        try:
            os.remove(save_path)
        except:
            pass

        return render_template('index.html', result=result)

    except Exception as e:
        app.logger.error(f"General Exception: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
