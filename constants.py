import os

# Base output directory
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
FACE_DATA_PATH = os.path.join(OUTPUT_DIR, 'face_cluster')
ENCODINGS_PATH = os.path.join(OUTPUT_DIR, 'encodings.pickle')
CLUSTERING_RESULT_PATH = OUTPUT_DIR

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)