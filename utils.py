import cv2
import math

def cv2_resize(image, target_width=256):
  h, w = image.shape[:2]
  image = cv2.resize(image, (target_width, math.floor(h/(w/target_width))), interpolation=cv2.INTER_CUBIC)
  return image