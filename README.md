🧠 Deepfake & AI-Generated Image Detector
A powerful deep learning-based web application that detects:

✅ Deepfake (face-manipulated) images
✅ AI-generated images (from models like DALL·E, GPT-4o, MidJourney)
✅ Real camera-captured images

Built using PyTorch, Flask, and ResNet18, this system uses two separate models trained on distinct datasets to intelligently analyze and classify uploaded images in real time.

🔍 Features
🧠 Dual-model architecture:
    • resnet18_face.pth: Detects face-swapped deepfakes
    • resnet18_art.pth: Detects AI-generated artistic images

🌐 Web-based interface with live upload and prediction

⚡ Fast, lightweight, and accurate

🛡️ Easily expandable into a unified 3-class classifier
