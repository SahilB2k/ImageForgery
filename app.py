

from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
import numpy as np
import cv2
import base64
from scipy.spatial.distance import cdist
from datetime import datetime

app = Flask(__name__)

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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/fake-detection")
def fake_detection_page():
    return render_template("fake_detection.html")

@app.route("/about-us")
def aboutUs():
    return render_template("aboutUs.html")

@app.route("/forgery-detection")
def forgery_detection_page():
    return render_template("forgery_detection.html")

@app.route("/predict", methods=["POST"])
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
        "analyzed_by": "SahilB2k"
    })

@app.route("/detect_forge", methods=["POST"])
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
        "analyzed_by": "SahilB2k"
    })

if __name__ == "__main__":
    app.run(debug=True)