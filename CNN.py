import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, datasets, models
from keras.models import Sequential

# --- Prepare Dataset ---
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize
train_images, test_images = train_images / 255.0, test_images / 255.0

# --- Create Model ---
model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])

# --- Compile Model ---
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# --- Train Model ---
epochs = 10
history = model.fit(train_images, train_labels, epochs=epochs)

# --- Visualize Training ---
acc = history.history['accuracy']
loss=history.history['loss']
plt.figure(figsize=(8, 8))
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), loss, label='Loss')
plt.legend(loc='lower right')
plt.title('Training Accuracy and Loss')

# --- Test Single Image ---
image = train_images[1].reshape(1,28,28,1)
prediction = model.predict(image, verbose=0)
predicted_class = np.argmax(prediction, axis=1)[0]

plt.imshow(image.reshape(28,28), cmap='gray')
print('Prediction of model:', predicted_class)

# --- Test Multiple Images ---
images = test_images[1:5]
plt.figure(figsize=(8,4))
for i, test_image in enumerate(images, start=1):
    org_image = test_image.reshape(28,28)
    test_image = test_image.reshape(1,28,28,1)
    prediction = model.predict(test_image, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]

    plt.subplot(1,4,i)
    plt.axis('off')
    plt.title(predicted_class)
    plt.imshow(org_image, cmap='gray')
plt.show()

# --- Save Model ---
model.save("tf-cnn-model.h5")

# --- Load Model and Test ---
loaded_model = models.load_model("tf-cnn-model.h5")
image = train_images[2].reshape(1,28,28,1)
prediction = loaded_model.predict(image, verbose=0)
predicted_class = np.argmax(prediction, axis=1)[0]

plt.imshow(image.reshape(28,28), cmap='gray')
print('Prediction of loaded model:', predicted_class)
