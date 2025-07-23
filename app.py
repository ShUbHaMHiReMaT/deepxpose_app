from flask import Flask, render_template, request
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import os

app = Flask(__name__)

# Common preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load face model
face_model = models.resnet18(weights=None)
face_model.fc = nn.Linear(face_model.fc.in_features, 2)
face_model.load_state_dict(torch.load("resnet18_face.pth", map_location=device))
face_model.eval()

# Load art model
art_model = models.resnet18(weights=None)
art_model.fc = nn.Linear(art_model.fc.in_features, 2)
art_model.load_state_dict(torch.load("resnet18_art.pth", map_location=device))
art_model.eval()

# Prediction logic
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        face_output = face_model(image)
        art_output = art_model(image)

    face_pred = torch.argmax(face_output, 1).item()
    art_pred = torch.argmax(art_output, 1).item()

    face_label = ["Fake (Deepfake)", "Real"][face_pred]
    art_label = ["AI Generated", "Real"][art_pred]

    return face_label, art_label

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    face_result = None
    art_result = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = "uploaded.jpg"
            file.save(file_path)

            face_result, art_result = predict(file_path)
            os.remove(file_path)

    return render_template("index.html", face_result=face_result, art_result=art_result)

if __name__ == "__main__":
    app.run(debug=True)
