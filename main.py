# Description:
# Euclidean Distance: A score of 0 is a perfect match. The score increases as images become less similar. 
# Your previous code correctly found the smallest distance.

# Cosine Similarity: A score of 1.0 is a perfect match (identical images). 
# The score decreases as images become less similar. 
# A score of 0 indicates no similarity (the vectors are perpendicular).

# ----------------------------------------------------------------------------

# Importing Required Packages
import torch
import os
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from glob import glob

# ---------------------------------------------------

# Loading the Model and the Feature Extractor
print("----------------Loading the Pre-trained ViT model and Feature Extractor------------------")
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
print("----------------Model Loaded Successfully----------------")

# ----------------------------------------------------

# Function to Extract Feature
def get_feature_vector(image_path):
    try:
        # Load the Image using pillow
        image = Image.open(image_path).convert("RGB")
        # Preprocess the image for the model
        inputs = feature_extractor(images=image, return_tensors="pt")
        # Get the model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract the feature vector (the output of the last hidden state)
        last_hidden_states = outputs.last_hidden_state
        feature_vector = last_hidden_states[:, 0, :].squeeze().numpy()
        return feature_vector
    except Exception as e:
        print(f"-------------------Error occurred while processing {image_path}: {e}----------------------------------")
        return None
    
# --------------------------------------------------------

# Function to create a feature vector database from existing images
def create_feature_database(database_folder):
    print(f"-----------------Creating Feature Database from {database_folder}---------------------------------")
    database = []
    # Loop through all images in the database folder
    image_paths = glob(os.path.join(database_folder, "*.jpg")) + glob(os.path.join(database_folder, "*.jpeg")) + glob(os.path.join(database_folder, "*.png"))
    for path in image_paths:
        print(f"--------- Processing {os.path.basename(path)}-----------------")
        feature_vector = get_feature_vector(path)
        if feature_vector is not None:
            database.append({'path':path, 'features': feature_vector})
    print(f"-----------------Feature Database Created Successfully. {len(database)} images processed-----------------")
    return database

# --------------------------------------------------------------------

# Function to Search and Match the database
def search_database(query_feature_vector, database, num_results = 1):
    print("----------------Searching the Database for Similar Images--------------------")
    # Extract feature vectors from the database for bulk processing
    db_feature_vectors = np.array([entry['features'] for entry in database])
    db_path = [entry['path'] for entry in database]
    # Calculate Euclidean distances
    distances = cosine_similarity(query_feature_vector.reshape(1, -1), db_feature_vectors)[0]
    # Combine Paths and distances, then sort by distances
    results = sorted(zip(db_path, distances), key=lambda x: x[1], reverse = True)
    return results[:num_results]

# --------------------------------------------------------------------

# The main Execution block
def find_closest_match(query_image_path):
    # Define the path to your existing database folder
    database_folder = r"C:\Users\Webbies\Jupyter_Notebooks\Rehou_Image_Search\Database"
    
    # Check if the query image exists
    if not os.path.exists(query_image_path):
        print(f"-------------------Error: Query image does not exist at {query_image_path}--------------------------")
        return None, None
        
    # Create the feature database
    feature_database = create_feature_database(database_folder)
    
    if not feature_database:
        print("No valid images found in the database folder. Please check folder path or image formats.")
        return None, None
        
    print(f"---------------------Processing the Query Image: {os.path.basename(query_image_path)}---------------------------")
    query_vector = get_feature_vector(query_image_path)
    
    if query_vector is not None:
        # Search the database for the single best match
        results = search_database(query_vector, feature_database, num_results=1)
        
        if results:
            # The search_database function returns a list of tuples, so we take the first element
            closest_match_path, distance = results[0]
            print(f"Match Found: {os.path.basename(closest_match_path)} with a distance of {distance:.4f}")
            return closest_match_path, distance
        else:
            print("No matches found.")
            return None, None
    else:
        print("------------------------The query vector is None-------------------------")
        return None, None