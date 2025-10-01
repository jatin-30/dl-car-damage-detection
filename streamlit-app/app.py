import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import os

# Class names
class_names = [
    "front_breakage",
    "front_crushed",
    "front_normal",
    "rear_breakage",
    "rear_crushed",
    "rear_normal"
]

# Lazy load model + transform (cached)
@st.cache_resource
def load_model():
    # local imports to avoid Streamlit's watcher touching torch at import time
    import torch
    import torch.nn as nn
    from torchvision import transforms, models

    # compute device inside function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model definition inside function to avoid top-level torchvision reference
    class CarClassifierResNet(nn.Module):
        def __init__(self, num_classes, dropout_rate=0.3):
            super().__init__()
            # use weights='DEFAULT' to follow newest torchvision API
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

    repo_root = Path(__file__).parent.parent.resolve()
    model_path = repo_root / "artifacts" / "saved_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = CarClassifierResNet(num_classes=len(class_names))
    state = torch.load(str(model_path), map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # build transform lazily and return it as well
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, device, transform

try:
    model, device, transform = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False
    model = None
    device = None
    transform = None

# Title
st.title("Car Damage Classifier")
st.write("Upload an image of a car and get the damage classification.")

# File uploader
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if not model_loaded:
        st.warning("Model is not loaded. Check the error message above.")
    else:
        # Preprocess -> ensure we use CPU if model is on CPU, else the chosen device
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict (import torch locally)
        import torch as _torch
        with _torch.no_grad():
            outputs = model(input_tensor)
            probs = _torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = class_names[np.argmax(probs)]

        # Output
        st.subheader(f"Predicted Class: `{predicted_class}`")
        st.write("Class probabilities:")
        for name, p in zip(class_names, probs):
            st.write(f"- {name}: {p:.4f}")
