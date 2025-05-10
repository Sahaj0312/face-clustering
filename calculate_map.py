import argparse
import cv2
import os
import pickle
import numpy as np
from mapcalc import calculate_map #, calculate_map_range
import random

from constants import OUTPUT_DIR


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def sort_detections(detections):
    return sorted(detections, key=lambda x: x["imagePath"])


def get_image(imagePath):
    image = cv2.imread(imagePath)
    return image, image.shape[1], image.shape[0]


def scale_bbox(bbox, w, h):
    if len(bbox) == 0:
        return bbox
    
    return bbox * np.array([w, h, w, h])


def draw_bboxes(image, bboxes, label, color=(0, 255, 0), thickness=20):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (int(x1), int(y1-20)), (int(x1+tw), int(y1)), color, -1)
        cv2.putText(image, label, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return image


def process_detections(ground_truth, predictions, output_folder, pred_label, iou_threshold=0.5):
    ground_truth = sort_detections(ground_truth)
    predictions = sort_detections(predictions)
    
    average_map = 0
    num_boxes = 0

    box_color = generate_random_color()

    with open(os.path.join(OUTPUT_DIR, f"map_{args["ground_truth"].split(".")[0]}_{args["predictions"].split(".")[0]}.txt"), "w") as fp:
        fp.write("imagePath,Average Precision\n")
        for gt, pred in zip(ground_truth, predictions):
            assert gt['imagePath'] == pred['imagePath'], "Image paths do not match"

            # image, w, h = get_image(os.path.join(output_folder, gt['imagePath'].split("/")[-1]))
            image, w, h = get_image(gt['imagePath'])

            gt_boxes = scale_bbox(np.array(gt['loc']), w, h)
            ground_truth_dict = {
                'boxes': gt_boxes,
                'labels': np.ones(len(gt['loc']))
            }
 
            image = draw_bboxes(image, gt_boxes, color=(0, 255, 0), label=labels['gt'])

            pred_boxes = scale_bbox(np.array(pred['loc']), w, h)
            prediction_dict = {
                'boxes': pred_boxes,
                'labels': np.ones(len(pred['loc']))
            }

            image = draw_bboxes(image, pred_boxes, color=box_color, label=pred_label)

            n_boxes = len(gt['loc'])

            avg_precision = 0.0
            if n_boxes > 0:
             avg_precision = calculate_map(ground_truth_dict, prediction_dict, iou_threshold)
            fp.write(f"{gt['imagePath']},{avg_precision:.6f}\n")

            average_map += avg_precision * n_boxes
            num_boxes += n_boxes

            cv2.imwrite(os.path.join(output_folder, os.path.basename(gt['imagePath'])), image)

    return average_map / num_boxes if num_boxes > 0 else 0


ap = argparse.ArgumentParser()
ap.add_argument("-gt", "--ground_truth", required=True,
    help="path to ground truth detections")
ap.add_argument("-p", "--predictions", required=True,
    help="path to predictions")
ap.add_argument("-o", "--output", required=True,
    help="path to output folder")
ap.add_argument("-l", "--label", required=True,
    help="label for predictions")
args = vars(ap.parse_args())

# Create output directory if it doesn't exist
os.makedirs(args["output"], exist_ok=True)

if __name__ == "__main__":
    print("[INFO] loading ground truth encodings...")
    ground_truth = pickle.loads(open(os.path.join(OUTPUT_DIR, args["ground_truth"]), "rb").read())
    # ground_truth = np.array(ground_truth)

    print("[INFO] loading predictions...")
    predictions = pickle.loads(open(os.path.join(OUTPUT_DIR, args["predictions"]), "rb").read())
    # predictions = np.array(predictions)

    labels = {"gt": "Insightface", "pred": "RetinaFace"}

    print("[INFO] Calculating MAP...")
    map_score = process_detections(ground_truth, predictions, args["output"], args["label"], iou_threshold=0.5)
    print("[INFO] MAP: {:.8f}".format(map_score))
