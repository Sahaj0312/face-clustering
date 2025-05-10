# USAGE
# python generate_auraface_embeddings.py -d detections.pickle -e encodings.pickle

import argparse
import cv2
import numpy as np
import pickle
import os
import time
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

from constants import OUTPUT_DIR, MODEL_DIR

# Set the model directory
# MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TARGET_SIZE = 640

def load_image(imagePath):
    """Load and convert image to RGB format."""
    image = cv2.imread(imagePath)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detections", required=True,
    help="path to serialized database of facial detections")
ap.add_argument("-e", "--encodings", required=True,
    help="path to output serialized database of facial encodings")
args = vars(ap.parse_args())

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the models
rec_model_path = os.path.join(MODEL_DIR, 'auraface/glintr100.onnx')
rec_model = model_zoo.get_model(rec_model_path)
rec_model.prepare(ctx_id=0, input_size=(TARGET_SIZE, TARGET_SIZE))

# Load detections
print("[INFO] loading detections...")
detections = pickle.loads(open(os.path.join(OUTPUT_DIR, args["detections"]), "rb").read())

data = []
with open(os.path.join(OUTPUT_DIR, "auraface_embedding_metadata.txt"), "w") as fp:
    fp.write("imagePath,num_faces,time_facial_encoding\n")
    
    # Process each image
    for i, detection in enumerate(detections):
        imagePath = detection["imagePath"]
        bboxes = np.array(detection["loc"])
        kpss = np.array(detection["kpss"])
        
        print(f"[INFO] processing image {i + 1}/{len(detections)}")
        print(imagePath)
        
        # Load and convert image
        image = load_image(imagePath)
        if image is None:
            print(f"[WARNING] Could not read image {imagePath}")
            continue

        # Preserve aspect ratio while resizing to target size
        h, w = image.shape[:2]
        scale = TARGET_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        
        embeddings = []
        start_time = time.time()
        
        # Process each face
        for bbox, kps in zip(bboxes, kpss):
            # Create Face object with bounding box
            x1, y1, x2, y2 = map(int, bbox[:4] * np.array([new_w, new_h, new_w, new_h]))  # Convert to int for indexing
            confidence = bbox[4] if len(bbox) > 4 else 1.0
            
            # Create Face object
            face = Face(bbox=bbox[:4], det_score=confidence, kps=kps * np.array([new_w, new_h]))
            
            # Get embedding
            rec_model.get(image, face)
            embeddings.append(face.normed_embedding)
            
        end_time = time.time()
        encoding_time = end_time - start_time
        print(f"[INFO] Facial encoding time: {encoding_time:.6f} seconds")
        
        # Write metadata
        fp.write(f"{imagePath},{len(bboxes)},{encoding_time}\n")
        
        # Save results
        for bbox, embedding in zip(bboxes, embeddings):
            d = {
                "imagePath": imagePath,
                "loc": bbox,
                "encoding": embedding
            }
            data.append(d)

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open(os.path.join(OUTPUT_DIR, args["encodings"]), "wb")
f.write(pickle.dumps(data))
f.close()
print("Encodings saved in {}".format(os.path.join(OUTPUT_DIR, args["encodings"])))
