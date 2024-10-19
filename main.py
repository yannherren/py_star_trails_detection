import cv2

import star_trails
from star_trails import detect_trails, unify_image_size, normalize_brightness

image = cv2.imread("output2.png", 0)

resized_image = unify_image_size(image)
normalized_image = normalize_brightness(resized_image)
print(detect_trails(normalized_image))