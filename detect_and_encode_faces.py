import argparse
import math
from imutils import paths
import torch
from torchvision import transforms
from edgeface.face_alignment import mtcnn
from edgeface.backbones import get_model
from PIL import Image
import pickle
import os
import time
import numpy as np

from constants import OUTPUT_DIR, TARGET_WIDTH, MODEL_DIR



mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))


model_name = "edgeface_s_gamma_05" # or edgeface_xs_gamma_06
model = get_model(model_name)
checkpoint_path=os.path.join(MODEL_DIR, f'{model_name}.pt')
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized database of facial encodings")
ap.add_argument("-d", "--detections", required=True,
	help="path to serialized database of facial detections")
args = vars(ap.parse_args())

def image_resize(image, target_width):
	width, height = image.size
	target_height = math.floor(height/(width/target_width))
	return image.resize((target_width, target_height))


def detect_faces(image):
	bboxes, faces = mtcnn_model.align_multi(image)

	return bboxes, faces

def encode_faces(faces):
	transformed_input = torch.stack([transform(face) for face in faces]) # preprocessing
	# extract embedding
	embeddings = model(transformed_input)

	return embeddings

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []

detections = []

print("Number of images: {}".format(len(imagePaths)))

with open(os.path.join(OUTPUT_DIR, f"edge_face_metadata_{TARGET_WIDTH}.txt"), "w") as fp:
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
			image = Image.open(imagePath).convert("RGB")

			# resize image
			if TARGET_WIDTH is not None:
				image = image_resize(image, TARGET_WIDTH)

			w, h = image.size

			# detect the (x, y)-coordinates of the bounding boxes
			# corresponding to each face in the input image
			start_time = time.time()
			bboxes, faces = detect_faces(image)
			end_time = time.time()
			face_detection_time = end_time - start_time
			print(f"[INFO] Face detection time: {face_detection_time:.6f} seconds")
			
			face_encoding_time = 0
			if len(faces) > 0:
				# compute the facial embedding for the face
				start_time = time.time()
				encodings = encode_faces(faces)
				end_time = time.time()
				facial_encoding_time = end_time - start_time
				print(f"[INFO] Facial encoding time: {facial_encoding_time:.6f} seconds")

			bboxes = np.array(bboxes)
			# save the detection result
			if len(bboxes) > 0:
				normalized_bboxes = bboxes.reshape(-1, 5)[:, :4] / np.array([[w, h, w, h]])
			else:
				normalized_bboxes = bboxes
			detections.append({"imagePath": imagePath, "loc": normalized_bboxes.tolist()})

			fp.write(f"{imagePath},{len(bboxes)},{face_detection_time},{facial_encoding_time}\n")

			# build a dictionary of the image path, bounding box location,
			# and facial encodings for the current image
			d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
					for (box, enc) in zip(bboxes.tolist(), encodings.detach().numpy())]
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
