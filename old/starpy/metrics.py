import cv2
import os

input_dir = "/Users/owen/starpipeline/classifer/data/solvable"

images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

def calculate_metrics(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Calculate the mean and standard deviation of the pixel values
    mean, stddev = cv2.meanStdDev(image)

    # Calculate the contrast (standard deviation of pixel values)
    contrast = stddev[0][0]

    # Calculate the brightness (mean of pixel values)
    brightness = mean[0][0]

    return contrast, brightness

for i in images:
    c, b = calculate_metrics(os.path.join(input_dir, i))
    print(f"Image: {i}, Contrast: {c}, Brightness: {b}")