import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Add parent directory to path so we can import from models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.efficientnetv2 import build_model

# Streamlit App Title
st.title("Lung Cancer Detection")

# File uploader
uploaded_file = st.file_uploader("Upload a lung X-ray image", type=["jpg", "jpeg", "png"])

# Load model and weights
@st.cache_resource
def load_model():
    model = build_model(num_classes=2)  # 2 classes: cancer and normal
    weights_path = os.path.join("weights", "efficientnetv2.pth")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        model.eval()
        return model
    else:
        st.error("Model weights not found. Please make sure 'weights/efficientnetv2.pth' exists.")
        return None

model = load_model()

# Inference
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    classes = ['Cancer', 'Normal']
    predicted_class = classes[predicted.item()]

    st.markdown(f"### Prediction: {predicted_class}")
