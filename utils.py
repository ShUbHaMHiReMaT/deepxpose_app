import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def load_models():
    # Build model architecture
    model_art = models.resnet18(weights=None)
    model_art.fc = torch.nn.Linear(model_art.fc.in_features, 2)
    model_art.load_state_dict(torch.load(r'D:\internship\resnet18_art.pth', map_location='cpu'))
    model_art.eval()

    model_face = models.resnet18(weights=None)
    model_face.fc = torch.nn.Linear(model_face.fc.in_features, 2)
    model_face.load_state_dict(torch.load(r'D:\internship\resnet18_face.pth', map_location='cpu'))
    model_face.eval()

    return model_art, model_face

def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return 'Fake' if predicted.item() == 1 else 'Real'
