# Face Clustering

A Python-based face clustering system that supports multiple face detection and encoding methods, followed by clustering similar faces using the DBSCAN algorithm. This project helps organize and categorize images containing faces by grouping similar-looking individuals together.

## Features

- Multiple face detection methods (InsightFace, RetinaFace, AuraFace)
- Automatic face detection and encoding
- Face clustering using DBSCAN
- Performance metrics and timing information
- Support for large-scale image processing

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn
- imutils
- InsightFace
- RetinaFace
- AuraFace

## Face Detection and Encoding Scripts

### 1. InsightFace Detection and Encoding

`insightface_detect_and_encode.py` uses the InsightFace library for both face detection and encoding.

```bash
python insightface_detect_and_encode.py \
    --dataset <path_to_folder_containing_images> \
    --encodings <output_encodings.pickle> \
    --detections <output_detections.pickle> \
    --model <path_to_model_directory>
```

Arguments:
- `--dataset`: Directory containing input images
- `--encodings`: Output file for facial encodings
- `--detections`: Output file for face detection results
- `--model`: Path to the InsightFace model directory

Features:
- Uses SCRFD for face detection
- Uses ArcFace for facial feature encoding
- Preserves aspect ratio during processing
- Outputs timing information for detection and encoding

This can be used to generate embeddings using both buffalo_l and AuraFace models.

You can doenload the buffalo_l model from [here](https://github.com/deepinsight/insightface/releases)

You can download the AuraFace model from [here](https://huggingface.co/fal/AuraFace-v1/tree/main)

### Model Names

| Model Name | Detection | Encoding |
| --- | --- | --- |
| buffalo_l | det_10g.onnx | w600k_r50.onnx |
| AuraFace | scrfd_10g_bnkps.onnx | glintr100 |


### 2. RetinaFace Detection

`retinaface_detection.py` specializes in face detection using the RetinaFace model.

```bash
python retinaface_detection.py \
    --dataset <path_to_images> \
    --detections <output_detections.pickle>
```

Arguments:
- `--dataset`: Directory containing input images
- `--detections`: Output file for face detection results

Features:
- High-accuracy face detection
- Includes confidence scores
- Extracts facial landmarks
- Normalizes bounding box coordinates

### 3. AuraFace Embeddings

`generate_auraface_embeddings.py` focuses on generating face embeddings using the AuraFace model.

```bash
python generate_auraface_embeddings.py \
    --detections <input_detections.pickle> \
    --encodings <output_encodings.pickle>
```

Arguments:
- `--detections`: Input file for face detection containing normalized bounding box locations
- `--encodings`: Output file for facial encodings

Features:
- Specialized in facial feature encoding
- Outputs normalized embeddings
- Includes processing time metrics
- Compatible with clustering pipeline

## Face Clustering

After generating face encodings using any of the above methods, use `cluster_faces.py` to group similar faces.
The script uses DBSCAN clustering algorithm to cluster the embeddings.

```bash
python cluster_faces.py \
    --encodings <path_to_encodings.pickle> \
    --label <output_label> \
    --jobs -1
```

Arguments:
- `--encodings`: Path to the facial encodings pickle file
- `--label`: Path to save clustering result labels
- `--jobs`: Number of parallel jobs (-1 for all CPUs)

## Performace Metrics

### 4. Calculate MAP

`calculate_map.py` provides a way to evaluate the performance of the face detection using the Mean Average Precision (MAP) metric.

The script additionally draws the predicted bounding boxes on the image and saves it in the output folder.

```bash
python calculate_map.py \
    --ground_truth <path_to_ground_truth.pickle> \
    --predictions <path_to_predictions.pickle>
    --output_folder <output_folder>
    --label <labels for the predicted bounding boxes>
    
```

Arguments:
- `--ground_truth`: Path to the ground truth detections pickle file
- `--predictions`: Path to the predicted detections pickle file
- `--output_folder`: Path to the output folder, where the images with predicted bounding boxes are saved.
- `--label`: Label for the predicted bounding boxes, printed on the image along with the bounding box.


### 5. Compare Clustering

`compare_clustering.py` helps compare the performance of different clustering outputs. This is particularly usefule when comparing model embeddings and their clusterability using DBSCAN.

It prints the Adjusted Rand Index (ARI) and the number of common images compared.
The Adjusted Rand Index (ARI) helps us measure how accurate a clustering result is by comparing it to the true labels (ground truth).
In our case, we assume clustering insightface embeddings give us the true labels.

It checks how well the pairs of points are grouped:
 - Are the same pairs together in both the real and predicted clusters?
 - Are different pairs also kept apart correctly?

```bash
python compare_clustering.py \
    --ground_truth <path_to_ground_truth.pickle> \
    --predictions <path_to_predictions.pickle>
```

Arguments:
- `--ground_truth`: Path to the ground truth clustering labels pickle file
- `--predictions`: Path to the predicted clustering labels pickle file


## Output Structure

```
output/
├── encodings/
│   ├── insightface_encodings.pickle
│   ├── retinaface_detections.pickle
│   └── auraface_encodings.pickle
├   └── auraface_clustering_labels.pickle
|
└── montages/
    ├── person_0_montage.jpg
    └── person_1_montage.jpg
```

## Performance Considerations

- InsightFace provides a good balance of accuracy and speed
- RetinaFace excels at detecting faces in challenging conditions
- AuraFace focuses on generating high-quality embeddings
- Processing time varies based on image size and hardware
- All methods support batch processing for better efficiency

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
