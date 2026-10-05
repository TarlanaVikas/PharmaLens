# extract_features.py

import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATASET_IMAGES_DIR = 'datasets/images'
DESCRIPTIONS_FILE = 'datasets/descriptions.csv'
FEATURES_FILE = 'features.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize the model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Updated to use 'weights' instead of 'pretrained'
model.eval()
# Remove the final classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract features
features = {}
for img_name in os.listdir(DATASET_IMAGES_DIR):
    img_path = os.path.join(DATASET_IMAGES_DIR, img_name)
    
    # Skip if it's a directory
    if not os.path.isfile(img_path):
        logger.warning(f"Skipping directory or non-file item: {img_name}")
        continue
    
    # Skip files with disallowed extensions
    if not allowed_file(img_name):
        logger.warning(f"Skipping non-image file: {img_name}")
        continue
    
    try:
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            feature = model(image).squeeze().numpy()
            feature = feature / np.linalg.norm(feature)  # Normalize
            features[img_name] = feature
        logger.info(f"Extracted features for {img_name}")
    except Exception as e:
        logger.error(f"Error processing {img_name}: {e}")

# Serialize features
with open(FEATURES_FILE, 'wb') as f:
    pickle.dump(features, f)

logger.info("Feature extraction completed and saved to features.pkl")
