import argparse
import pickle
import numpy as np
from sklearn.metrics import adjusted_rand_score
from constants import CLUSTERING_RESULT_PATH
import os

def load_labels(pickle_path):
    """Load labels from a pickle file and return a dictionary mapping image paths to labels"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return {item['imagePath']: item['label'] for item in data}

def calculate_accuracy(gt_labels, pred_labels):
    """Calculate clustering accuracy using adjusted rand index"""
    # Get common image paths
    common_paths = set(gt_labels.keys()) & set(pred_labels.keys())
    if not common_paths:
        raise ValueError("No common images found between ground truth and predictions")
    
    # Convert to lists ensuring same order
    paths = sorted(list(common_paths))
    gt = [gt_labels[p] for p in paths]
    pred = [pred_labels[p] for p in paths]
    
    # Calculate adjusted rand index
    ari = adjusted_rand_score(gt, pred)
    return ari, len(common_paths), len(gt_labels), len(pred_labels)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare clustering labels')
    parser.add_argument('-gt', '--ground_truth', required=True, help='Path to ground truth labels pickle file')
    parser.add_argument('-p', '--predictions', required=True, help='Path to predicted labels pickle file')
    args = parser.parse_args()

    # Load labels
    print("[INFO] Loading ground truth labels...")
    gt_path = os.path.join(CLUSTERING_RESULT_PATH, args.ground_truth)
    gt_labels = load_labels(gt_path)

    print("[INFO] Loading predicted labels...")
    pred_path = os.path.join(CLUSTERING_RESULT_PATH, args.predictions)
    pred_labels = load_labels(pred_path)

    # Calculate and print accuracy
    try:
        ari, common_count, gt_count, pred_count = calculate_accuracy(gt_labels, pred_labels)
        print(f"\n[RESULTS]")
        print(f"Total images in ground truth: {gt_count}")
        print(f"Total images in predictions: {pred_count}")
        print(f"Common images compared: {common_count}")
        print(f"Adjusted Rand Index: {ari:.4f}")
    except ValueError as e:
        print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    main()
