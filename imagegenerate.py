import os
import cv2
import numpy as np
from keras.datasets import mnist

# --- Folder to save ideal digit images ---
output_folder = "digits"
os.makedirs(output_folder, exist_ok=True)

# --- Load MNIST dataset from Keras ---
(x_train, y_train), _ = mnist.load_data()

# --- Save a fixed number of images per digit ---
count_per_digit = 1
digit_counts = {i: 0 for i in range(10)}

for img, label in zip(x_train, y_train):
    if digit_counts[label] >= count_per_digit:
        continue

    # MNIST is black digit on white; invert to match your model input
    img_inverted = cv2.bitwise_not(img)

    filename = f"{label}_{digit_counts[label]}.png"
    cv2.imwrite(os.path.join(output_folder, filename), img_inverted)

    digit_counts[label] += 1

    if all(count >= count_per_digit for count in digit_counts.values()):
        break

print(f"Generated {count_per_digit*10} MNIST-style images in '{output_folder}' folder.")
