# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
# insightface library by @insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
# argument parser
import argparse
# pickle to save the encodings
import pickle
# openCV
import cv2
# operating system
import os
import numpy as np

from constants import OUTPUT_DIR, MODEL_DIR

import time

# Model expects input size of 640x640
TARGET_SIZE = 640

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
    help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized database of facial encodings")
ap.add_argument("-d", "--detections", required=True,
    help="path to serialized database of facial detections")
ap.add_argument("-m", "--model", required=True,
    help="name of the model")

args = vars(ap.parse_args())

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

det_model_file = f'{args["model"]}/scrfd_10g_bnkps.onnx' # det_10g.onnx
rec_model_file = f'{args["model"]}/glintr100.onnx' # w600k_r50.onnx

# Initialize the models
det_model_path = os.path.join(MODEL_DIR, det_model_file)
rec_model_path = os.path.join(MODEL_DIR, rec_model_file)

det_model = model_zoo.get_model(det_model_path)
rec_model = model_zoo.get_model(rec_model_path)

det_model.prepare(ctx_id=0, input_size=(TARGET_SIZE, TARGET_SIZE), det_thres=0.5)
rec_model.prepare(ctx_id=0, input_size=(TARGET_SIZE, TARGET_SIZE))

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
detections = []

with open(os.path.join(OUTPUT_DIR, f"{args['model']}_metadata_{TARGET_SIZE}.txt"), "w") as fp:
    fp.write("imagePath,num_faces,time_face_detection,time_facial_encoding\n")
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # load the input image
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        print(imagePath)

        if imagePath.endswith(".jpg"):
            # loading image
            image = cv2.imread(imagePath)
            if image is None:
                print(f"[WARNING] Could not read image {imagePath}")
                continue

            # convert to RGB (dlib needs RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preserve aspect ratio while resizing to target size
            h, w = image.shape[:2]
            scale = TARGET_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized_image = cv2.resize(image, (new_w, new_h))

            # Create a square canvas of TARGET_SIZE x TARGET_SIZE
            canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
            # Calculate position to paste the image
            y_offset = (TARGET_SIZE - new_h) // 2
            x_offset = (TARGET_SIZE - new_w) // 2
            # Paste the image
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

            # detect the (x, y)-coordinates of the bounding boxes
            start_time = time.time()
            bboxes, kpss = det_model.detect(canvas, max_num=0, metric='default')
            end_time = time.time()
            face_detection_time = end_time - start_time
            print(f"[INFO] Face detection time: {face_detection_time:.6f} seconds")
            
            if bboxes is not None:
                # Adjust bounding boxes back to original image coordinates
                scale_x = w / new_w
                scale_y = h / new_h
                adjusted_bboxes = []
                adjusted_kpss = []
                
                for bbox, kps in zip(bboxes, kpss):
                    # Adjust bbox coordinates
                    x1 = (bbox[0] - x_offset) * scale_x
                    y1 = (bbox[1] - y_offset) * scale_y
                    x2 = (bbox[2] - x_offset) * scale_x
                    y2 = (bbox[3] - y_offset) * scale_y
                    adjusted_bbox = np.array([x1, y1, x2, y2])
                    adjusted_bboxes.append(adjusted_bbox)
                    
                    # Adjust keypoints
                    adjusted_kp = kps.copy()
                    adjusted_kp[:, 0] = (kps[:, 0] - x_offset) * scale_x
                    adjusted_kp[:, 1] = (kps[:, 1] - y_offset) * scale_y
                    adjusted_kpss.append(adjusted_kp)
                
                bboxes = np.array(adjusted_bboxes)
                kpss = np.array(adjusted_kpss)
            
                embeddings = []
                start_time = time.time()
                for bbox, kps in zip(bboxes, kpss):
                    face = Face(bbox=bbox, kps=kps, det_score=bbox[4] if len(bbox) > 4 else 1.0)
                    rec_model.get(image, face)
                    embeddings.append(face.normed_embedding)
                end_time = time.time()
                facial_encoding_time = end_time - start_time
                print(f"[INFO] Facial encoding time: {facial_encoding_time:.6f} seconds")
                fp.write(f"{imagePath},{len(bboxes)},{face_detection_time},{facial_encoding_time}\n")

                # save the detection result
                if bboxes.shape[0] > 0:
                    normalized_bboxes = bboxes / np.array([[w, h, w, h]])
                else:
                    normalized_bboxes = bboxes
                detections.append({"imagePath": imagePath, "loc": normalized_bboxes.tolist()})

                # build a dictionary of the image path, bounding box location,
                # and facial encodings for the current image
                d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
                    for (box, enc) in zip(bboxes, embeddings)]
                data.extend(d)

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open(os.path.join(OUTPUT_DIR, args["encodings"]), "wb")
f.write(pickle.dumps(data))
f.close()
print("Encodings of images saved in {}".format(os.path.join(OUTPUT_DIR, args["encodings"])))

# dump the facial detections data to disk
print("[INFO] serializing detections...")
f = open(os.path.join(OUTPUT_DIR, args["detections"]), "wb")
f.write(pickle.dumps(detections))
f.close()
print("Detections of images saved in {}".format(os.path.join(OUTPUT_DIR, args["detections"])))
