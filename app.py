from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
import numpy as np
import cv2
import base64
from scipy.spatial.distance import cdist
import pyrebase
import os
from functools import wraps
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management
app.permanent_session_lifetime = timedelta(days=5)

# Firebase Configuration
firebase_config = {
    "apiKey": "AIzaSyAnWscCH3Jao-8lqaNbNCleQ675UB_SoO0",
    "authDomain": "imageforgery-1dc9f.firebaseapp.com",
    "databaseURL": "https://placeholder-database-url.firebaseio.com",
    "projectId": "imageforgery-1dc9f",
    "storageBucket": "imageforgery-1dc9f.appspot.com",
    "messagingSenderId": "589646432154",
    "appId": "1:589646432154:web:8e5b547de6a42a52ed1c23"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

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

# Image processing functions
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

# Load model and initialize detector
model = Discriminator()
try:
    model.load_state_dict(torch.load("discriminator_model_flask.pth", map_location=torch.device("cpu")))
except Exception as e:
    print(f"Error loading model: {e}")
model.eval()

detector = CopyMoveDetector(model)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)
# Routes
@app.route("/login", methods=["GET", "POST"])
def login():
    current_time = "2025-04-18 10:20:54"  # Updated current time
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        username = request.form.get("username")  # In case it's passed from signup
        
        try:
            # Authenticate with Firebase
            user = auth.sign_in_with_email_and_password(email, password)
            
            # Set session data
            session.permanent = True
            session['user'] = user['localId']
            session['username'] = username if username else email.split('@')[0]  # Use username if provided, otherwise use email prefix
            session['token'] = user['idToken']
            
            # Set username in localStorage for welcome message
            return '''
                <script>
                    localStorage.setItem('userName', '{}');
                    window.location.href = '{}';
                </script>
            '''.format(session['username'], url_for('home'))
            
        except Exception as e:
            return render_template("login.html", 
                                error="Invalid email or password. Please try again.",
                                current_time=current_time,
                                username="SahilB2k")
    
    return render_template("login.html", 
                         current_time=current_time,
                         username="SahilB2k")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    current_time = "2025-04-18 10:20:54"  # Updated current time
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if password != confirm_password:
            return render_template("signup.html", 
                                error="Passwords do not match.",
                                current_time=current_time,
                                username="SahilB2k")
        
        try:
            # Create user in Firebase
            user = auth.create_user_with_email_and_password(email, password)
            auth.send_email_verification(user['idToken'])
            
            # Pass username to login page
            return render_template("login.html", 
                                message="Account created! Please verify your email and login.",
                                username=username,
                                current_time=current_time)
        except Exception as e:
            error_message = str(e)
            if "EMAIL_EXISTS" in error_message:
                return render_template("signup.html", 
                                    error="Email already exists.",
                                    current_time=current_time,
                                    username="SahilB2k")
            return render_template("signup.html", 
                                error=f"Error creating account: {error_message}",
                                current_time=current_time,
                                username="SahilB2k")
    
    return render_template("signup.html", 
                         current_time=current_time,
                         username="SahilB2k")

@app.route("/")
def home():
    """Home page route"""
    current_time = "2025-04-18 10:20:54"  # Updated current time
    return render_template("index.html", 
                         logged_in='user' in session,
                         username=session.get('username', 'User'),
                         current_time=current_time)

@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    current_time = "2025-04-18 10:20:54"  # Updated current time
    if request.method == "POST":
        email = request.form.get("email")
        
        try:
            # Send password reset email through Firebase
            auth.send_password_reset_email(email)
            return render_template("login.html", 
                                message="Password reset email sent! Please check your inbox.",
                                current_time=current_time,
                                username="SahilB2k")
        except Exception as e:
            return render_template("reset_password.html", 
                                error="Error sending reset email. Please try again.",
                                current_time=current_time,
                                username="SahilB2k")
    
    return render_template("reset_password.html", 
                         current_time=current_time,
                         username="SahilB2k")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route("/about-us")
def aboutUs():
    return render_template("aboutUs.html", 
                         logged_in='user' in session,
                         username=session.get('username', None))

@app.route("/fake-detection")
def fake_detection_page():
    """Fake detection page - no login required"""
    return render_template("fake_detection.html", 
                         logged_in='user' in session,
                         username=session.get('username', None))

@app.route("/forgery-detection")
@login_required
def forgery_detection_page():
    """Forgery detection page - login required"""
    return render_template("forgery_detection.html", 
                         logged_in=True,
                         username=session.get('username', None))

@app.route("/predict", methods=["POST"])
def predict():
    """Handle fake image detection"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image_bytes = file.read()
        input_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            output_tensor, _ = model(input_tensor)
            confidence = output_tensor.mean().item()

        prediction = "Fake" if confidence > 0.5 else "Real"
        
        return jsonify({
            "prediction": prediction, 
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect_forge", methods=["POST"])
@login_required
def detect_forge():
    """Handle forgery detection - login required"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image_bytes = file.read()
        
        input_tensor = preprocess_image(image_bytes)
        forgery_map = detector.generate_forgery_map(input_tensor)
        visualization = apply_forgery_heatmap(image_bytes, forgery_map)
        
        buffered = io.BytesIO()
        visualization.save(buffered, format="PNG")
        visualization_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        with torch.no_grad():
            output, _ = model(input_tensor)
            confidence = output.mean().item()
        
        affected_area = float(np.mean(forgery_map > 0.5) * 100)
        num_regions = len(np.unique(forgery_map > 0.5)) - 1
        
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            "prediction": "Fake" if confidence > 0.5 else "Real",
            "confidence": float(confidence),
            "heatmap": visualization_base64,
            "affected_area_percentage": affected_area,
            "num_copied_regions": int(num_regions),
            "timestamp": current_time,
            "analyzed_by": session.get('username', 'Unknown user')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', 
                         logged_in='user' in session,
                         username=session.get('username', None)), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html', 
                         logged_in='user' in session,
                         username=session.get('username', None)), 500

if __name__ == "__main__":
    app.run(debug=True)