from flask import Flask, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import cv2
import numpy as np
import os
import requests
import logging
import random

# Initialize Flask app
app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///luxury_images.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define the database model
class ImageAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    format = db.Column(db.String(20))
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    blurry = db.Column(db.String(10))
    prediction = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    caption = db.Column(db.Text)
    feedback = db.Column(db.Text)

# Create the database (run once)
with app.app_context():
    db.create_all()

# Run the app

# === CONFIG ===
CUSTOM_VISION_PREDICTION_KEY = "7j4giuRUiS8V4ycv52wfMoCkPYyBZXUrvTXaW00P2mLrRghP7lpRJQQJ99BFACYeBjFXJ3w3AAAIACOGg5aY"
CUSTOM_VISION_ENDPOINT = "https://project4823-prediction.cognitiveservices.azure.com/"
CUSTOM_VISION_PROJECT_ID = "961c6cf3-471c-423b-9726-2b9bea7f75f3"
CUSTOM_VISION_PUBLISHED_NAME = "Iteration2"

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

def generate_feedback(prediction, caption, blurry):
    feedback_parts = []

    # Blurriness feedback
    if blurry == 'Yes':
        blurry_feedback_options = [
            "The image seems blurry. A clearer photo might yield better results.",
            "Blurriness detected—consider retaking the image for improved analysis.",
            "This image is not very sharp. Try uploading a higher-quality version."
        ]
        feedback_parts.append(random.choice(blurry_feedback_options))

    # Prediction feedback
    prediction = prediction.lower()
    if prediction == "luxury":
        luxury_feedback_options = [
            "This image is classified as luxury—great choice!",
            "Looks luxurious! The model picked up high-end features.",
            "Luxury detected. This image has premium characteristics."
        ]
        feedback_parts.append(random.choice(luxury_feedback_options))
    elif prediction == "not luxury":
        not_luxury_feedback_options = [
            "This image is classified as not luxury. Consider enhancing its appeal.",
            "Not luxury detected. You might improve lighting or composition.",
            "The image seems basic. Try adding more premium elements."
        ]
        feedback_parts.append(random.choice(not_luxury_feedback_options))
    elif prediction == "miscellaneous":
        misc_feedback_options = [
            "The image falls under miscellaneous. Try uploading a more specific subject.",
            "Miscellaneous category detected—consider refining the image content.",
            "Uncategorized image. A clearer subject might help classification."
        ]
        feedback_parts.append(random.choice(misc_feedback_options))

    # Caption-based feedback
    if caption:
        caption_lower = caption.lower()

        # Vehicles
        if any(vehicle in caption_lower for vehicle in ["luxury car", "sports car", "yacht", "private jet", "boat", "flight", "sports bike"]):
            feedback_parts.append("This image features a luxury vehicle. It adds a premium touch.")
        elif any(vehicle in caption_lower for vehicle in ["bicycle", "scooter", "motorcycle", "bus", "train"]):
            feedback_parts.append("This image shows a basic mode of transport. Consider showcasing more upscale options.")

        # Food
        if any(food in caption_lower for food in ["pizza", "burger", "pasta", "chips", "snacks"]):
            feedback_parts.append("The image contains junk food. For a luxury impression, consider gourmet or fine dining visuals.")

        # Spaces
        if "interior" in caption_lower:
            feedback_parts.append("The image appears to depict an interior space. Consider emphasizing design elements for a luxury feel.")
        elif "outdoor" in caption_lower or "landscape" in caption_lower:
            feedback_parts.append("Outdoor scenes can be enhanced with better lighting and composition.")
        elif "furniture" in caption_lower:
            feedback_parts.append("Furniture-focused images benefit from clean backgrounds and good lighting.")
        elif "room" in caption_lower:
            feedback_parts.append("Room images should highlight spaciousness and decor for a luxury impression.")
        elif "bathroom" in caption_lower:
            feedback_parts.append("Bathroom images should emphasize cleanliness and high-end fixtures.")
        elif "building" in caption_lower:
            feedback_parts.append("Building images should highlight architectural details and luxury materials.")
        elif "phone" in caption_lower:
            feedback_parts.append("Phone images should emphasize sleek design and premium features.")
        elif "watch" in caption_lower:
            feedback_parts.append("Watch images should focus on craftsmanship and luxury branding.")
        elif "jewelry" in caption_lower:
            feedback_parts.append("Jewelry images should highlight intricate details and luxury materials.")
        elif "clothing" in caption_lower:
            feedback_parts.append("Clothing images should focus on fabric quality and stylish presentation.")
        elif "shoes" in caption_lower:
            feedback_parts.append("Shoe images should emphasize design and premium materials.")
        elif "accessories" in caption_lower:
            feedback_parts.append("Accessory images should highlight unique designs and luxury branding.")
        elif "electronics" in caption_lower:
            feedback_parts.append("Electronics images should focus on innovative features and sleek designs.")
        elif "art" in caption_lower:
            feedback_parts.append("Art images should emphasize creativity and high-quality presentation.")
        elif "decoration" in caption_lower:
            feedback_parts.append("Decoration images should highlight unique styles and luxury aesthetics.")
        elif "garden" in caption_lower:
            feedback_parts.append("Garden images should focus on lush greenery and elegant landscaping.")
        elif "pool" in caption_lower:
            feedback_parts.append("Pool images should emphasize luxury features and serene surroundings.")
        elif "spa" in caption_lower:
            feedback_parts.append("Spa images should focus on relaxation and high-end amenities.")
        elif "hotel" in caption_lower:
            feedback_parts.append("Hotel images should highlight luxury accommodations and exceptional service.")
        elif "restaurant" in caption_lower:
            feedback_parts.append("Restaurant images should focus on elegant dining settings and gourmet presentations.")
        elif "resort" in caption_lower:
            feedback_parts.append("Resort images should highlight luxury amenities and beautiful surroundings.")
        elif "yacht" in caption_lower:
            feedback_parts.append("Yacht images should emphasize elegance and premium features.")
        elif "private jet" in caption_lower:
            feedback_parts.append("Private jet images should highlight luxury interiors and exclusive amenities.")
        elif "car" in caption_lower:
            feedback_parts.append("Car images should emphasize sleek design and high-performance features.")
        elif "bike" in caption_lower:
            feedback_parts.append("Bike images should focus on stylish design and premium materials.")
        else:
            feedback_parts.append(f"Caption insight: '{caption}'.")

    return " ".join(feedback_parts)

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
        feedback_text = generate_feedback(prediction_tag, description, 'Yes' if blurry else 'No')
        result['feedback'] = feedback_text

        new_entry = ImageAnalysis(
        filename=image_file.filename,
        format=result['format'],
        width=result['width'],
        height=result['height'],
        blurry=result['blurry'],
        prediction=result['prediction'],
        confidence=result['prediction_confidence'],
        caption=result.get('caption', ''),
        feedback=result.get('feedback', '')
        )
        db.session.add(new_entry)
        db.session.commit()

        # Clean up uploaded file
        try:
            os.remove(save_path)
        except:
            pass

        return render_template('index.html', result=result)
    

    except Exception as e:
        app.logger.error(f"General Exception: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

from sqlalchemy import or_, func

@app.route('/records')
def records():
    search_query = request.args.get('search', '').strip().lower()
    filter_blurry = request.args.get('blurry', '')

    query = ImageAnalysis.query

    if search_query:
        query = query.filter(
            or_(
                ImageAnalysis.filename.ilike(f'%{search_query}%'),
                ImageAnalysis.caption.ilike(f'%{search_query}%'),
                func.lower(ImageAnalysis.prediction) == search_query  # exact match only
            )
        )

    if filter_blurry in ['Yes', 'No']:
        query = query.filter_by(blurry=filter_blurry)

    all_records = query.all()
    return render_template('records.html', records=all_records, search=search_query, blurry_filter=filter_blurry)

if __name__ == '__main__':
    app.run(debug=True)
