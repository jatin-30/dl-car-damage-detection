import streamlit as st
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = [
    "front_breakage",
    "front_crushed",
    "front_normal",
    "rear_breakage",
    "rear_crushed",
    "rear_normal"
]


# Load model definition
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)



DEFAULT_MODEL_REL_PATH = Path("artifacts") / "saved_model.pth"
@st.cache_resource
def load_model():
    # Resolve path relative to this file (works on Streamlit Cloud)
    base_dir = Path(__file__).parent.resolve()
    model_path = (base_dir / DEFAULT_MODEL_REL_PATH).resolve()

    # Debug info for Streamlit logs / UI
    st.write(f"Base dir: {base_dir}")
    st.write(f"Looking for model at: {model_path}")

    if not model_path.exists():
        # helpful debug listing
        st.error("Model file not found at the expected path.")
        st.write("Files in repo root:", sorted([p.name for p in base_dir.iterdir() if p.exists()]))
        if (base_dir / "artifacts").exists():
            st.write("Files in artifacts:", sorted([p.name for p in (base_dir / "artifacts").iterdir()]))
        # raise a helpful error (Streamlit will show it)
        raise FileNotFoundError(f"No such file: {model_path}")

    # instantiate model architecture
    model = CarClassifierResNet(num_classes=len(class_names))

    # load using CPU/mapped device first
    state = torch.load(str(model_path), map_location="cpu")
    # if you saved full model vs state_dict, handle both:
    if isinstance(state, dict) and "state_dict" in state and not any(k.startswith("__") for k in state):
        # if saved as {'state_dict': ...}
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # if saved as full model object (less recommended), try load_state_dict, else load directly
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # fallback: maybe the file contains a full model object
        try:
            model = state
        except Exception as e:
            raise RuntimeError("Failed to load model state_dict or model object.") from e

    # move to device and eval
    model.to(device)
    model.eval()
    return model

# call it
model = load_model()


# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Title
st.title("Car Damage Classifier")
st.write("Upload an image of a car and get the damage classification.")

# File uploader
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class = class_names[np.argmax(probs)]

    # Output
    st.subheader(f"Predicted Class: `{predicted_class}`")
