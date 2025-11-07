from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import io
import os

from model import DeepfakeDetector
from utils import preprocess_image, predict

app = Flask(__name__)
CORS(app)  # Allow frontend to call API

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepfakeDetector(num_classes=2).to(device)

# Load trained weights
model_path = 'best_model.pt'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
else:
    print(f"❌ Model file not found: {model_path}")

@app.route('/')
def home():
    return jsonify({
        'message': 'Deepfake Detection API',
        'status': 'running',
        'model_accuracy': '94.74%',
        'endpoints': {
            '/predict': 'POST - Upload image for prediction',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists(model_path),
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        # Check if image is in request - FIXED: changed 'image' to 'file'
        if 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['file']  # FIXED: changed 'image' to 'file'
        
        # Check if file is valid
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess
        img_tensor = preprocess_image(image)
        
        # Predict
        result = predict(model, img_tensor, device)
        
        # FIXED: Return proper format for frontend
        # The HTML expects 'prediction' and 'confidence'
        return jsonify({
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)