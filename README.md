# Car Damage Classification using CNN & Transfer Learning

A deep learning project for classifying vehicle damage types using images. This model can detect 6 types of car damage (or lack thereof) from front and rear perspectives using a CNN or fine-tuned pretrained models like ResNet and EfficientNet.

---

## Model Objective

Given an image of a car (front or rear view), the model predicts one of the following 6 classes:

- `front_breakage`
- `front_crushed`
- `front_normal`
- `rear_breakage`
- `rear_crushed`
- `rear_normal`

This can be used in insurance automation, damage assessment, or fleet management systems.

---

## Features

- Custom CNN, ResNet-50, and EfficientNet-B0 backbones
- Transfer learning with fine-tuning
- Hyperparameter tuning using Optuna
- Streamlit-powered interactive frontend for image upload & prediction

---
## Dataset

The full dataset is **not included** in this repository due to size constraints.

The expected folder structure for the dataset is:
1. front_breakage
1. front_crushed
1. front_normal
1. rear_breakage
1. rear_crushed
1. rear_normal

You can add your own images following this structure to train or test the model.

For quick testing, a few sample images are included in the `sample-data` folder.


---
## Streamlit Web App

### How to Run

```bash
# Install requirements
pip install -r requirements.txt

# Launch the app
streamlit run app.py
