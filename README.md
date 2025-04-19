# Face Clustering

A Python-based face clustering system that groups similar faces together using facial encodings and the DBSCAN clustering algorithm. This project helps organize and categorize images containing faces by grouping similar-looking individuals together.

## Features

- Automatically clusters similar faces from a collection of images
- Creates organized folders for each unique person detected
- Generates montage images showing all faces in each cluster
- Handles outlier faces that don't match any clusters
- Supports parallel processing for improved performance

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn
- imutils
- pickle

## Installation

1. Clone this repository:

```bash
cd Face-Clustering
```

2. Install the required dependencies:

```bash
pip install opencv-python numpy scikit-learn imutils
```

## Project Structure

```
Face-Clustering/
├── cluster_faces.py     # Main clustering script
├── constants.py         # Project constants and paths
├── encodings.pickle     # Facial encodings database (generated)
└── output/             # Clustering results
    ├── person_0/       # Images of person 0
    ├── person_1/       # Images of person 1
    ├── ...
    ├── person_0_montage.jpg  # Montage of person 0's faces
    ├── person_1_montage.jpg  # Montage of person 1's faces
    └── unknown_faces_montage.jpg  # Montage of unmatched faces
```

## Usage

The face clustering process involves two main steps:

### Step 1: Generate Face Encodings

First, you need to generate facial encodings from your image dataset using the `encode_faces.py` script:

```bash
python encode_faces.py --dataset path/to/your/dataset --encodings output/encodings.pickle --detection_method "cnn"
```

Arguments:

- `--dataset`: Path to your input dataset of images
- `--encodings`: Path where the facial encodings will be saved
- `--detection_method`: Face detection method to use ("cnn" for better accuracy, "hog" for faster processing)

This step will:

1. Scan through all images in your dataset
2. Detect faces in each image
3. Generate 128-dimensional facial encodings
4. Save all encodings to a pickle file

### Step 2: Cluster the Faces

Once you have generated the encodings, run the clustering script:

```bash
python cluster_faces.py --encodings output/encodings.pickle --jobs -1
```

Arguments:

- `--encodings`: Path to the serialized facial encodings pickle file (required)
- `--jobs`: Number of parallel jobs to run (-1 uses all CPUs, default: -1)

## Output

The script will:

1. Create separate folders for each unique person detected (`person_0`, `person_1`, etc.)
2. Save individual images in their respective person folders
3. Generate montage images showing all faces for each person
4. Create a special montage for faces that couldn't be clustered (outliers)

## How It Works

1. The system loads pre-computed facial encodings from a pickle file
2. Uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to cluster similar faces
3. Groups faces based on their 128-dimensional facial encodings
4. Organizes images into folders and creates visual montages
5. Handles outlier faces that don't match any clusters

## Notes

- The quality of clustering depends on the quality of input images and facial encodings
- Adjust DBSCAN parameters if needed for better clustering results
- The system works best with clear, front-facing facial images
- Processing time depends on the number of images and CPU cores available

## License

[Add your license information here]

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
