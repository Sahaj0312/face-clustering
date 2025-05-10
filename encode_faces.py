# USAGE
	# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
# face_recognition library by @ageitgey
import face_recognition
# argument parser
import argparse
# pickle to save the encodings
import pickle
# openCV
import cv2
# operating system
import os

from constants import OUTPUT_DIR, TARGET_WIDTH
from utils import cv2_resize

import time



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized database of facial encodings")
ap.add_argument("-dt", "--detections", required=True,
	help="path to serialized database of facial detections")
ap.add_argument("-d", "--detection_method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []

detections = []

with open(os.path.join(OUTPUT_DIR, f"face_recognition_metadata_{TARGET_WIDTH}.txt"), "w") as fp:
	fp.write("imagePath,num_faces,time_face_detection, time_facial_encoding\n")
	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# load the input image and convert it from RGB (OpenCV ordering)
		# to dlib ordering (RGB)
		print("[INFO] processing image {}/{}".format(i + 1,
				len(imagePaths)))
		print(imagePath)

		if imagePath.endswith(".jpg"):
			# loading image to BGR
			image = cv2.imread(imagePath)

			# ocnverting image to RGB format
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			# resize image
			if TARGET_WIDTH is not None:
				image = cv2_resize(image, TARGET_WIDTH)

			h, w, _ = image.shape

			# detect the (x, y)-coordinates of the bounding boxes
			# corresponding to each face in the input image
			start_time = time.time()
			boxes = face_recognition.face_locations(image,
					model=args["detection_method"])
			end_time = time.time()
			face_detection_time = end_time - start_time
			print(f"[INFO] Face detection time: {face_detection_time:.6f} seconds")
			

			# compute the facial embedding for the face
			start_time = time.time()
			encodings = face_recognition.face_encodings(image, boxes)
			end_time = time.time()
			facial_encoding_time = end_time - start_time
			print(f"[INFO] Facial encoding time: {facial_encoding_time:.6f} seconds")
			fp.write(f"{imagePath},{len(boxes)},{face_detection_time},{facial_encoding_time}\n")

			# save the detection result
			detections.append({"imagePath": imagePath, "loc": 
			[[float(box[3]/w), float(box[0]/h), float(box[1]/w), float(box[2]/h)] for box in boxes]})

			# build a dictionary of the image path, bounding box location,
			# and facial encodings for the current image
			d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
					for (box, enc) in zip(boxes, encodings)]
			data.extend(d)

# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open(os.path.join(OUTPUT_DIR, args["encodings"]), "wb")
f.write(pickle.dumps(data))
f.close()
print("Encodings of images saved in {}".format(os.path.join(OUTPUT_DIR, args["encodings"])))

# dump the detection results to disk
print("[INFO] serializing detections...")
f = open(os.path.join(OUTPUT_DIR, args["detections"]), "wb")
f.write(pickle.dumps(detections))
f.close()
print("Detections of images saved in {}".format(os.path.join(OUTPUT_DIR, args["detections"])))
