# app.py

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
DATASET_IMAGES_DIR = 'datasets/images'
DESCRIPTIONS_FILE = 'datasets/descriptions.csv'
FEATURES_FILE = 'features.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
TOP_K = 10  # Number of top matches to return

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secure_secret_key_here'  # Replace with a strong, random secret key

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load dataset features
with open(FEATURES_FILE, 'rb') as f:
    dataset_features = pickle.load(f)
logger.info(f"Loaded features for {len(dataset_features)} images.")

# Load descriptions
descriptions_df = pd.read_csv(DESCRIPTIONS_FILE)
descriptions_df['medicine_name'] = descriptions_df['medicine_name'].str.lower()
logger.info(f"Loaded descriptions for {len(descriptions_df)} images.")

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

def extract_feature(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        feature = model(image).squeeze().numpy()
        feature = feature / np.linalg.norm(feature)  # Normalize
    return feature

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'medicine_image' not in request.files:
            logger.warning("No file part in the request.")
            return render_template('index.html', error='No file part')
        file = request.files['medicine_image']
        # If user does not select file, browser may submit an empty part without filename
        if file.filename == '':
            logger.warning("No selected file.")
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            logger.info(f"Uploaded file saved to {upload_path}")

            # Extract feature from uploaded image
            try:
                input_feature = extract_feature(upload_path)
                logger.info(f"Extracted feature from uploaded image {filename}")
            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
                return render_template('index.html', error='Error processing image.')

            # Compute similarities
            dataset_image_filenames = list(dataset_features.keys())
            dataset_matrix = np.array([dataset_features[name] for name in dataset_image_filenames])
            input_vector = input_feature.reshape(1, -1)
            similarities = cosine_similarity(input_vector, dataset_matrix)[0]

            # Find top K matches
            top_indices = similarities.argsort()[::-1][:TOP_K]
            top_matches = []
            for idx in top_indices:
                matched_image = dataset_image_filenames[idx]
                similarity = similarities[idx]
                # Threshold for similarity can be set as needed
                if similarity < 0.7:
                    logger.warning(f"Similarity {similarity} for image {matched_image} is below threshold.")
                    continue  # Skip if similarity is too low

                # Retrieve description and other details
                desc_row = descriptions_df[descriptions_df['image_filename'] == matched_image]
                if not desc_row.empty:
                    description = desc_row.iloc[0]['description']
                    uses = desc_row.iloc[0].get('uses', 'N/A')
                    side_effects = desc_row.iloc[0].get('side_effects', 'N/A')
                    dosage = desc_row.iloc[0].get('dosage', 'N/A')
                    manufacturer = desc_row.iloc[0].get('manufacturer', 'N/A')
                    contraindications = desc_row.iloc[0].get('contraindications', 'N/A')
                    storage_instructions = desc_row.iloc[0].get('storage_instructions', 'N/A')
                    medicine_name = desc_row.iloc[0]['medicine_name'].title()
                else:
                    logger.warning(f"No description found for image {matched_image}.")
                    description = "Description not available."
                    uses = "N/A"
                    side_effects = "N/A"
                    dosage = "N/A"
                    manufacturer = "N/A"
                    contraindications = "N/A"
                    storage_instructions = "N/A"
                    medicine_name = "Unknown Medicine"

                top_matches.append({
                    'medicine_name': medicine_name,
                    'description': description,
                    'uses': uses,
                    'side_effects': side_effects,
                    'dosage': dosage,
                    'manufacturer': manufacturer,
                    'contraindications': contraindications,
                  'storage_instructions': storage_instructions,
                    'image_filename': matched_image,
                    'similarity': float(similarity)  # Convert to Python float
                })
                logger.info(f"Match found: {medicine_name} with similarity {similarity}")

            if top_matches:
                # Store top matches in session for confirmation
                session['top_matches'] = top_matches
                session['uploaded_image'] = filename
                return redirect(url_for('confirm_match'))
            else:
                logger.warning("No matching medicine found above similarity threshold.")
                return render_template('index.html', error='No matching medicine found.')

    return render_template('index.html')

@app.route('/confirm', methods=['GET', 'POST'])
def confirm_match():
    top_matches = session.get('top_matches', None)
    uploaded_image = session.get('uploaded_image', None)

    if not top_matches or not uploaded_image:
        logger.warning("No top matches or uploaded image found in session.")
        return redirect(url_for('upload_file'))

    if request.method == 'POST':
        # User has confirmed the match
        selected_medicine = request.form.get('selected_medicine')
        if selected_medicine:
            # Retrieve the medicine details
            medicine_details = next((item for item in top_matches if item['medicine_name'] == selected_medicine), None)
            if medicine_details:
                logger.info(f"User confirmed medicine: {selected_medicine}")
                return render_template('medicine_details.html', medicine=medicine_details, uploaded_image=uploaded_image)
        # If no selection made
        logger.warning("User submitted confirmation without selecting a medicine.")
        return render_template('confirm_match.html', error='No medicine selected.', top_matches=top_matches, uploaded_image=uploaded_image)

    return render_template('confirm_match.html', top_matches=top_matches, uploaded_image=uploaded_image)

# Route to serve dataset images
@app.route('/dataset_images/<filename>')
def dataset_image(filename):
    return send_from_directory(os.path.join(app.root_path, DATASET_IMAGES_DIR), filename)

if __name__ == '__main__':
    app.run(debug=True)
