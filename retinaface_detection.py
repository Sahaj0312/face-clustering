# USAGE
# python retinaface_detection.py --dataset dataset --detections detections.pickle

# import the necessary packages
from imutils import paths
from retinaface import RetinaFace
import numpy as np
import argparse
import pickle
import cv2
import os
import time

from utils import cv2_resize


from constants import OUTPUT_DIR, TARGET_WIDTH

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help="path to input directory of faces + images")
ap.add_argument("-d", "--detections", required=True,
    help="path to serialized database of facial detections")
args = vars(ap.parse_args())

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize RetinaFace
detector = RetinaFace.build_model()

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
detections = []

with open(os.path.join(OUTPUT_DIR, f"retinaface_metadata_{TARGET_WIDTH}.txt"), "w") as fp:
    fp.write("imagePath,num_faces,time_face_detection\n")
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        print(imagePath)

        if imagePath.endswith(".jpg"):
            # loading image
            image = cv2.imread(imagePath)
            if image is None:
                print(f"[WARNING] Could not read image {imagePath}")
                continue

            # convert to RGB (RetinaFace expects RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if TARGET_WIDTH is not None:
                image = cv2_resize(image, TARGET_WIDTH)

            h, w, _ = image.shape

            # detect faces
            start_time = time.time()
            resp = RetinaFace.detect_faces(image, model=detector)
            end_time = time.time()
            face_detection_time = end_time - start_time
            print(f"[INFO] Face detection time: {face_detection_time:.6f} seconds")

            if isinstance(resp, dict):
                # Extract bounding boxes
                bboxes = []
                confidences = []
                kpss = []
                for face_idx in resp:
                    face_info = resp[face_idx]
                    bbox = face_info["facial_area"]  # [x1, y1, x2, y2]
                    confidence = face_info["score"]
                    kps = [kp for _, kp in face_info["landmarks"].items()]
                    # Add confidence score to bbox
                    # bbox.append(confidence)
                    bboxes.append(bbox)
                    confidences.append(confidence)
                    kpss.append(kps)
                    

                bboxes = np.array(bboxes)
                kpss = np.array(kpss)
                if len(bboxes) > 0:
                    bboxes = bboxes / np.array([w, h, w, h])
                    kpss = kpss / np.array([[[w, h]]])

                print(f"[INFO] Found {len(bboxes)} faces")
                fp.write(f"{imagePath},{len(bboxes)},{face_detection_time}\n")

                # save the detection result
                detections.append({
                    "imagePath": imagePath,
                    "loc": bboxes.tolist(),
                    "conf": confidences,
                    "kpss": kpss.tolist()
                })
            else:
                print(f"[WARNING] No faces detected in {imagePath}")
                fp.write(f"{imagePath},0,{face_detection_time}\n")

# dump the facial detections data to disk
print("[INFO] serializing detections...")
f = open(os.path.join(OUTPUT_DIR, args["detections"]), "wb")
f.write(pickle.dumps(detections))
f.close()
print("Detections of images saved in {}".format(os.path.join(OUTPUT_DIR, args["detections"])))
