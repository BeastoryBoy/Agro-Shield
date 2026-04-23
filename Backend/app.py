import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torchvision.transforms.functional as TF
from torchvision import models
import torch.nn as nn

# 🔥 DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 🔥 INIT APP
app = Flask(__name__)
CORS(app)

# 📊 LOAD DATA
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# 🔥 CORRECT CLASS ORDER (VERY IMPORTANT)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background',
    'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca',
    'Grape___Leaf_blight',
    'Grape___healthy',
    'Pepper___Bacterial_spot',
    'Pepper___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato__Target_Spot',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Mosaic_virus',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites',
    'Tomato___Yellow_leaf_curl_virus',
    'Tomato___healthy'
]

# 🤖 LOAD MODEL (MATCH TRAINING)
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

state_dict = torch.load("plant_disease_model_1_latest.pt", map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# 🔍 PREDICTION FUNCTION
def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    input_data = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_data)

    probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

    confidence = float(np.max(probs))
    index = int(np.argmax(probs))

    return index, confidence, probs


# 🌐 ROUTES
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


# 🚀 MAIN API
@app.route('/submit', methods=['POST'])
def submit():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    image = request.files['image']
    filename = image.filename

    os.makedirs('static/uploads', exist_ok=True)
    file_path = os.path.join('static/uploads', filename)
    image.save(file_path)

    # 🔥 PREDICTION
    pred, confidence, probs = prediction(file_path)

    # 🔥 CLASS NAME FIX (MAIN FIX)
    title = class_names[pred]

    # 🔥 CONFIDENCE GAP CHECK
    sorted_probs = np.sort(probs)
    confidence_gap = sorted_probs[-1] - sorted_probs[-2]

    # 🔥 NON-LEAF FILTER
    if title == "Background" or confidence < 0.70 or confidence_gap < 0.10:
        return jsonify({
            "isPlant": False,
            "errorTitle": "No Leaf Detected",
            "errorMessage": "This image does not appear to contain a clear plant leaf.",
            "errorSuggestion": "Upload a clear, close-up image of a single leaf."
        })

    # 🔥 CLEAN NAME FOR UI
    clean_name = title.replace("___", " ").replace("__", " ").replace("_", " ")

    # 🔥 HEALTH CHECK
    is_healthy = "healthy" in title.lower()

    # 📊 MATCH CSV DATA (SAFE LOOKUP)
    row = disease_info[disease_info['disease_name'] == title]

    if not row.empty:
        description = row.iloc[0]['description']
        prevent = row.iloc[0]['Possible Steps']
    else:
        description = "No description available"
        prevent = "Follow general plant care practices"

    # 🌿 ORGANIC LOGIC
    name_lower = title.lower()

    if is_healthy:
        organic_option = ""
    elif "blight" in name_lower:
        organic_option = "Use neem oil or compost tea spray"
    elif "bacterial" in name_lower:
        organic_option = "Use copper-based organic spray"
    elif "mildew" in name_lower:
        organic_option = "Use baking soda spray"
    elif "rust" in name_lower:
        organic_option = "Use sulfur-based organic fungicide"
    elif "virus" in name_lower:
        organic_option = "Remove infected leaves and control insects"
    elif "mite" in name_lower:
        organic_option = "Use neem oil or insecticidal soap"
    else:
        organic_option = "Use general organic fungicide"

    # 🔥 SEVERITY LOGIC
    if is_healthy:
        severity = "none"
        urgency = "none"
    else:
        if confidence > 0.90:
            severity = "low"
            urgency = "routine"
        elif confidence > 0.80:
            severity = "med"
            urgency = "moderate"
        else:
            severity = "high"
            urgency = "immediate"

    spread = "Low" if is_healthy else "Moderate"
    impact = "None" if is_healthy else "Depends on severity"

    # 🚀 FINAL RESPONSE
    return jsonify({
        "isPlant": True,
        "plantName": clean_name.split(" ")[0],
        "scientificName": "",
        "isHealthy": is_healthy,
        "condition": clean_name,
        "diseaseType": "Healthy" if is_healthy else "Disease",
        "severity": severity,
        "confidence": int(confidence * 100),
        "affectedParts": ["Leaf"],
        "symptoms": [description[:120]],
        "cause": description,
        "spreadRisk": spread,
        "yieldImpact": impact,
        "urgency": urgency,
        "treatment": [] if is_healthy else [prevent],
        "prevention": [prevent],
        "organicOption": organic_option
    })


# 🚀 RUN
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)