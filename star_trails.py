import cv2
import numpy as np
import math

IMAGE_PROCESS_WIDTH_PX = 1000
NORMALIZED_BRIGHTNESS = 100

def detect_trails(normalized_image):
    gamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(normalized_image, table)
    edges = cv2.Canny(gamma_corrected, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, minLineLength=10, maxLineGap=2)

    angles = []
    line_lengths = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if line is None:
                continue

            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            line_length = math.hypot(x2 - x1, y2 - y1)
            angles.append(abs(angle))
            line_lengths.append(line_length)

    angles = [0] if len(angles) == 0 else angles
    line_lengths = [0] if len(line_lengths) == 0 else line_lengths

    return np.median(angles), np.median(line_lengths)

def unify_image_size(image):
    img_height, img_width = image.shape[:2]
    aspect_ratio = img_height / img_width
    resized_image = cv2.resize(image, (IMAGE_PROCESS_WIDTH_PX, int(IMAGE_PROCESS_WIDTH_PX * aspect_ratio)))

    return resized_image

def normalize_brightness(image):
    brightness = np.median(image)
    delta = NORMALIZED_BRIGHTNESS - brightness
    normalized_image = cv2.convertScaleAbs(image, alpha=1, beta=delta)

    return normalized_image

def calculate_compensation(trail_length, trail_angle, original_image, focal_length_mm, sensor_width_mm, sensor_height_mm):
    width, height = original_image.shape[:2]
    size_reduction_factor = width / IMAGE_PROCESS_WIDTH_PX
    x_length_px = math.cos(trail_angle) * trail_length * size_reduction_factor
    y_length_px = math.sin(trail_angle) * trail_length * size_reduction_factor

    fov_h_rad = 2 * math.atan2(sensor_width_mm, 2 * focal_length_mm)
    fov_v_rad = 2 * math.atan2(sensor_height_mm, 2 * focal_length_mm)

    fov_h = math.degrees(fov_h_rad)
    fov_v = math.degrees(fov_v_rad)

    degree_per_px_h = fov_h / width
    degree_per_px_v = fov_v / height

    x_compensation_deg = x_length_px * degree_per_px_h
    y_compensation_deg = y_length_px * degree_per_px_v

    return x_compensation_deg, y_compensation_deg
