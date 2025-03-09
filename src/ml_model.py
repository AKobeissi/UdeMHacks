import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import streamlit as st
from config import MODEL_PATH, PARASITE_CLASSES

@st.cache_resource
def load_model():
    """Load the parasite detection model"""
    model = resnet50(pretrained=True)
    num_classes = len(PARASITE_CLASSES)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load trained weights if available
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            st.warning(f"Error loading model weights: {str(e)}. Using pretrained ResNet50.")
    else:
        st.warning(f"Model weights file not found at {MODEL_PATH}. Using pretrained ResNet50.")
    
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess an image for the model"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)), #originally 224 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image):
    """Make a prediction with the model"""
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted[0]].item()
    
    return PARASITE_CLASSES[predicted[0]], confidence

def analyze_sample(image):
    """Analyze a sample image and return diagnosis and confidence"""
    # Load model with map_location to handle GPU-trained models on CPU
    model = torch.load(
        r'C:\Users\akobe\OneDrive\UdeMHacks\src\complete_parasite_model.pt', 
        map_location=torch.device('cpu')
    )
    processed_image = preprocess_image(image)
    diagnosis, confidence = predict(model, processed_image)
    return diagnosis, confidence