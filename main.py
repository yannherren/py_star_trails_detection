import cv2

import star_trails
from star_trails import detect_trails, unify_image_size, normalize_brightness, calculate_compensation

image = cv2.imread("demo_data/output.png", 0)

resized_image = unify_image_size(image)
normalized_image = normalize_brightness(resized_image)

angle, length = detect_trails(normalized_image)

yaw, pitch = calculate_compensation(length, angle, image, 250, 22.3, 14.9)

print(yaw * 15) # 15 = gear ratio
print(pitch * 15)
