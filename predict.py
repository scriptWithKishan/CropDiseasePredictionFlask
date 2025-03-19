import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image



# Configuration
MODEL_PATH = "./crop_disease_detection.h5"
UPLOAD_FOLDER = "temp_uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = (224, 224)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Load Class Names
try:
CLASSES = [
    "Apple___Scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_(Two-spotted_spider_mite)",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

    print(f"✅ Manually loaded classes: {CLASSES}")
    print(f"✅ Found classes: {CLASSES}")
except Exception as e:
    print(f"❌ Error loading classes: {e}")
    CLASSES = []

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper Functions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Routes
@app.route("/")
def home():
    return "Welcome to the Crop Disease Prediction API. Use POST /predict."

@app.route('/predict', methods=['POST'])
def predict():
    filepath = None  # Ensure variable is defined for cleanup
    if 'image' not in request.files:
        app.logger.error("No image uploaded in request")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if not file or not allowed_file(file.filename):
        app.logger.error("Invalid file type or missing file")
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Save the uploaded file safely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.info(f"Saved image to {filepath}")

        # Preprocess the image for prediction
        img_array = preprocess_image(filepath)
        app.logger.info(f"Preprocessed Image Shape: {img_array.shape}")

        # Ensure classes are loaded
        if not CLASSES:
            raise ValueError("No classes loaded. Check your dataset path.")

        # Perform model prediction
        print("Starting Prediction...")  # Debug print
        predictions = model.predict(img_array)
        print("Prediction Completed:", predictions)  # Debug print

        class_idx = np.argmax(predictions[0])
        result = {
            "class": CLASSES[class_idx],
            "confidence": float(np.max(predictions[0]))
        }
        app.logger.info(f"Prediction successful: {result}")
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            app.logger.info(f"Removed temporary file: {filepath}")

if __name__ == "__main__":
    app.run(debug=True)

