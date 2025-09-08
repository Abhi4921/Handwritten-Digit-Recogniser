import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# --- Load your trained CNN model ---
model_path = "tf-cnn-model.h5"  # path to your saved model
model = load_model(model_path)

# --- Folder containing your handwritten digit images ---
image_folder = "my_digits"

# --- Function to preprocess a real-world digit ---
def preprocess_digit(image_path):
    # 1. Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    # 2. Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5,5), 0)

    # 3. Adaptive threshold (white digit on black background)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 4. Find contours and biggest bounding box
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"No digit found in image: {image_path}")
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = img[y:y+h, x:x+w]

    # 5. Resize while keeping aspect ratio to 20x20
    if w > h:
        new_w = 20
        new_h = max(1, int(h * (20 / w)))
    else:
        new_h = 20
        new_w = max(1, int(w * (20 / h)))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 6. Pad to 28x28
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    digit_padded = np.pad(digit_resized, ((pad_top, pad_bottom),(pad_left,pad_right)), "constant", constant_values=0)

    # 7. Normalize
    digit_normalized = digit_padded.astype("float32") / 255.0

    # 8. Reshape for CNN input
    return digit_normalized.reshape(1, 28, 28, 1)

# --- Loop through images and predict ---
plt.figure(figsize=(12, 4))
for i, filename in enumerate(os.listdir(image_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, filename)
    processed_img = preprocess_digit(img_path)
    if processed_img is None:
        continue

    # Predict
    prediction = model.predict(processed_img, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]

    print(f"Image: {filename} → Predicted digit: {predicted_class}")

    # Display image with prediction
    plt.subplot(1, len(os.listdir(image_folder)), i+1)
    plt.imshow(processed_img.reshape(28,28), cmap='gray')
    plt.title(predicted_class)
    plt.axis('off')

plt.tight_layout()
plt.show()
