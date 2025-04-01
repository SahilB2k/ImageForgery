from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn

app = Flask(__name__)  # Removed template_folder override



# Define the Discriminator model
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
        return self.main(x)

# Load the model
model = Discriminator()
model.load_state_dict(torch.load("discriminator_model_flask.pth", map_location=torch.device("cpu")))
model.eval()


##################################################################

# Add at the beginning
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()  # Binary cross-entropy loss

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    image_id = data.get("image_id")
    true_label = data.get("true_label")  # "Real" or "Fake" from user feedback
    
    # Retrieve the image and prediction from storage (you'd need to implement this)
    stored_image = retrieve_image(image_id)
    input_tensor = preprocess_image(stored_image)
    
    # Set to training mode
    model.train()
    
    # Forward pass
    output = model(input_tensor)
    
    # Convert label to tensor (1 for Fake, 0 for Real)
    target = torch.tensor([[1.0]]) if true_label == "Fake" else torch.tensor([[0.0]])
    
    # Compute loss
    loss = criterion(output.mean(), target)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save updated model periodically
    torch.save(model.state_dict(), "discriminator_model_flask.pth")
    
    # Set back to evaluation mode for future predictions
    model.eval()
    
    return jsonify({"status": "Model updated", "loss": float(loss.item())})

###################################################################################################################

# Define image preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjusted for RGB images
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

@app.route("/")
def home():
    return render_template("index.html")  # Make sure this file exists in templates folder

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    input_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        # This outputs a 1xCxHxW tensor where C=1 for your model
        output_tensor = model(input_tensor)
        
        # Get the average prediction value (properly handling the tensor dimensions)
        confidence = output_tensor.mean().item()

    # Since your Discriminator outputs a probability, classify it
    prediction = "Fake" if confidence > 0.5 else "Real"

    return jsonify({"prediction": prediction, "confidence": float(confidence)})

if __name__ == "__main__":
    app.run(debug=True)