import argparse
import cv2
from imutils import paths
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import pickle

from constants import OUTPUT_DIR, TARGET_WIDTH, MODEL_DIR
from utils import cv2_resize


def get_bbox(detection):
    # Draw bounding_box
    bbox = detection.bounding_box
    left, top = bbox.origin_x, bbox.origin_y
    right, bottom = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

    return [float(left), float(top), float(right), float(bottom)]


# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, 'blaze_face_short_range.tflite'))
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--detections", required=True,
	help="path to serialized database of facial encodings")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []

detections = []

with open(os.path.join(OUTPUT_DIR, f"mp_face_detection_metadata_{TARGET_WIDTH}.txt"), "w") as fp:
	fp.write("imagePath,num_faces,time_face_detection\n")
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

			# STEP 3: Load the input image.
			image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

			# STEP 4: Detect faces in the input image.
			start_time = time.time()
			detection_result = detector.detect(image)
			end_time = time.time()
			face_detection_time = end_time - start_time
			print(f"[INFO] Face detection time: {face_detection_time:.6f} seconds")


			# STEP 5: Process the detection result.
			bboxes = []
			if detection_result.detections:
				num_faces = len(detection_result.detections)
				for detection in detection_result.detections:
					bboxes.append(get_bbox(detection))
			else:
				num_faces = 0

			# write metadata
			fp.write(f"{imagePath},{num_faces},{face_detection_time}\n")

			# save the detection result
			detections.append({"imagePath": imagePath, "loc": bboxes})

# dump the detection results to disk
print("[INFO] serializing detections...")
f = open(os.path.join(OUTPUT_DIR, args["detections"]), "wb")
f.write(pickle.dumps(detections))
f.close()
print("Detections of images saved in {}".format(os.path.join(OUTPUT_DIR, args["detections"])))
