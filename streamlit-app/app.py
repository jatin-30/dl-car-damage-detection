import streamlit as st
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


# Load trained model
@st.cache_resource
def load_model():
    model = CarClassifierResNet(num_classes=len(class_names))
    model.load_state_dict(torch.load("saved_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


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
