# USAGE
# python cluster_faces.py --encodings encodings.pickle

# import the necessary packages
# DBSCAN model for clustering similar encodings
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import shutil
import os
import math
from constants import FACE_DATA_PATH, ENCODINGS_PATH, CLUSTERING_RESULT_PATH

# add constants file in the code (clustering_result)

def move_image(image, id, labelID):
	# Create the label directory inside output
	path = os.path.join(CLUSTERING_RESULT_PATH, f'person_{labelID}')
	os.makedirs(path, exist_ok=True)
	
	filename = f'face_{id}.jpg'
	# Using cv2.imwrite() method 
	# Saving the image 
	
	cv2.imwrite(os.path.join(path, filename), image)
	
	return

def create_montage(faces, labelID):
	if len(faces) == 0:
		return None
		
	# Calculate grid size based on number of faces
	grid_size = math.ceil(math.sqrt(len(faces)))
	montage = build_montages(faces, (96, 96), (grid_size, grid_size))[0]
	
	# Save montage
	title = f"person_{labelID}_montage"
	title = "unknown_faces_montage" if labelID == -1 else title
	cv2.imwrite(os.path.join(CLUSTERING_RESULT_PATH, f'{title}.jpg'), montage)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of parallel jobs to run (-1 will use all CPUs)")
args = vars(ap.parse_args())

# Create output directory if it doesn't exist
os.makedirs(CLUSTERING_RESULT_PATH, exist_ok=True)

# load the serialized face encodings + bounding box locations from
# disk/encodings pickle file, then extract the set of encodings to so we can cluster on them

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# cluster the embeddings
print("[INFO] clustering...")

# creating DBSCAN object for clustering the encodings with the metric "euclidean"
clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"])
clt.fit(encodings)

# determine the total number of unique faces found in the dataset
# clt.labels_ contains the label ID for all faces in our dataset (i.e., which cluster each face belongs to).
# To find the unique faces/unique label IDs, used NumPy's unique function.
# The result is a list of unique labelIDs
labelIDs = np.unique(clt.labels_)

# we count the numUniqueFaces . There could potentially be a value of -1 in labelIDs — this value corresponds
# to the "outlier" class where a 128-d embedding was too far away from any other clusters to be added to it.
# "outliers" could either be worth examining or simply discarding based on the application of face clustering.
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
for labelID in labelIDs:
	print("[INFO] processing faces for person {}".format(labelID))
	idxs = np.where(clt.labels_ == labelID)[0]
	
	# initialize the list of faces to include in the montage
	faces = []
	
	# loop over all indexes
	for i in idxs:
		# load the input image and extract the face ROI
		image = cv2.imread(data[i]["imagePath"])
		(top, right, bottom, left) = data[i]["loc"]
		face = image[top:bottom, left:right]
		
		# move the full image to the appropriate person folder
		move_image(image, i, labelID)
		
		# resize the face ROI and add it to the faces list for montage
		face = cv2.resize(face, (96, 96))
		faces.append(face)
	
	# create and save the montage
	create_montage(faces, labelID)

print("[INFO] Processing complete. Results saved in the 'output' directory.")
print("[INFO] Each person's images are saved in separate folders named 'person_0', 'person_1', etc.")
print("[INFO] Montages for each person are saved as 'person_X_montage.jpg'")