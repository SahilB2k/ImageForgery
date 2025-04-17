from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
import torch
from flask_cors import CORS
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
import numpy as np
import cv2
from dotenv import load_dotenv
import base64
from datetime import datetime
import os

# Initialize Flask app with secret key
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/verify-token": {
        "origins": ["http://localhost:5000"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})
app.secret_key = os.environ.get('SECRET_KEY') # Change this in production

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate(r"C:\Users\sahil jadhav\Downloads\serviceAccountKey.json.json")
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Firebase initialization error: {e}")

# First Discriminator for fake detection
class FakeDetector(nn.Module):
    def __init__(self):
        super(FakeDetector, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Second Discriminator for forgery detection
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.main):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return x, features

# Keep your existing CopyMoveDetector class
class CopyMoveDetector:
    def __init__(self, model):
        self.model = model
        self.gradients = []
        self.features = None
        
    def save_gradient(self, grad):
        self.gradients.append(grad)

    def detect_copied_regions(self, feature_maps, threshold=0.85):
        B, C, H, W = feature_maps.shape
        features_reshaped = feature_maps.view(C, H * W).t()
        
        similarity = torch.mm(features_reshaped, features_reshaped.t())
        similarity = similarity / torch.norm(features_reshaped, dim=1).unsqueeze(0)
        similarity = similarity / torch.norm(features_reshaped, dim=1).unsqueeze(1)
        
        mask = (similarity > threshold).float()
        mask = mask.view(H, W, H, W)
        
        heatmap = torch.sum(mask, dim=(2, 3))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.detach().cpu().numpy()

    def generate_forgery_map(self, input_tensor):
        self.gradients = []
        self.features = None
        
        output, features = self.model(input_tensor)
        feature_maps = features[-2]
        
        copy_move_map = self.detect_copied_regions(feature_maps)
        
        feature_maps.requires_grad_(True)
        feature_maps.register_hook(self.save_gradient)
        self.features = feature_maps
        
        pred = output.mean()
        
        self.model.zero_grad()
        pred.backward()
        
        if len(self.gradients) > 0:
            gradients = self.gradients[0]
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            
            for i in range(self.features.shape[1]):
                self.features[:, i, :, :] *= pooled_gradients[i]
                
            activation_map = torch.mean(self.features, dim=1).squeeze()
            activation_map = torch.relu(activation_map)
            
            if torch.max(activation_map) > 0:
                activation_map = activation_map / torch.max(activation_map)
            
            combined_map = 0.7 * copy_move_map + 0.3 * activation_map.detach().cpu().numpy()
            combined_map = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min() + 1e-8)
            
            return combined_map
        
        return copy_move_map

def apply_forgery_heatmap(image_bytes, heatmap):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image)
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    threshold = 0.5
    binary_mask = (heatmap_resized > threshold).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
    
    overlay = img_array.copy()
    alpha = 0.4
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 50:
            mask = (labels == i).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
            
            region_color = heatmap_color * mask
            overlay = (1 - alpha * mask) * overlay + alpha * region_color
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    
    return Image.fromarray(overlay)

# Load models
fake_detector_model = FakeDetector()
forgery_detector_model = Discriminator()

try:
    fake_detector_model.load_state_dict(torch.load("discriminator_model_flask.pth", map_location=torch.device("cpu")))
    forgery_detector_model.load_state_dict(torch.load("discriminator_model_flask.pth", map_location=torch.device("cpu")))
except Exception as e:
    print(f"Error loading models: {e}")

fake_detector_model.eval()
forgery_detector_model.eval()

# Initialize forgery detector
detector = CopyMoveDetector(forgery_detector_model)

# Authentication middleware
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/login')
def login():
    return render_template('Login.html')  # Make sure it matches your template name case

@app.route('/sign-up')
def signup():
    return render_template('signup.html')

@app.route('/verify-token', methods=['POST'])
def verify_token():
    try:
        data = request.get_json()
        if not data or 'idToken' not in data:
            return jsonify({'success': False, 'error': 'No token provided'}), 400

        id_token = data['idToken']
        
        try:
            # Verify the token
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            email = decoded_token.get('email', '')
            name = data.get('name') or decoded_token.get('name', '')

            # Store user info in session
            session['user_id'] = uid
            session['email'] = email
            session['name'] = name
            
            return jsonify({
                'success': True,
                'user': {
                    'uid': uid,
                    'email': email,
                    'name': name
                }
            })
        except auth.InvalidIdTokenError:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route("/about-us")
def aboutUs():
    return render_template("aboutUs.html")

@app.route("/fake-detection")
@login_required  # Protected route
def fake_detection_page():
    return render_template("fake_detection.html")

@app.route("/forgery-detection")
@login_required  # Protected route
def forgery_detection_page():
    return render_template("forgery_detection.html")

@app.route("/predict", methods=["POST"])
@login_required  # Protected route
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    input_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        output = fake_detector_model(input_tensor)
        confidence = output.mean().item()

    return jsonify({
        "prediction": "Fake" if confidence > 0.5 else "Real",
        "confidence": float(confidence),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "analyzed_by": session.get('email', 'Unknown')  # Use session email
    })

@app.route("/detect_forge", methods=["POST"])
@login_required  # Protected route
def detect_forge():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    
    input_tensor = preprocess_image(image_bytes)
    forgery_map = detector.generate_forgery_map(input_tensor)
    visualization = apply_forgery_heatmap(image_bytes, forgery_map)
    
    buffered = io.BytesIO()
    visualization.save(buffered, format="PNG")
    visualization_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    with torch.no_grad():
        output, _ = forgery_detector_model(input_tensor)
        confidence = output.mean().item()
    
    affected_area = float(np.mean(forgery_map > 0.5) * 100)
    num_regions = len(np.unique(forgery_map > 0.5)) - 1
    
    return jsonify({
        "prediction": "Fake" if confidence > 0.5 else "Real",
        "confidence": float(confidence),
        "heatmap": visualization_base64,
        "affected_area_percentage": affected_area,
        "num_copied_regions": int(num_regions),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "analyzed_by": session.get('email', 'Unknown')  # Use session email
    })

if __name__ == "__main__":
    app.run(debug=True)