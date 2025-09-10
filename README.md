# Handwritten-Digit-Recogniser
* This project demonstrates a handwritten digit recognition system using the MNIST dataset and a Convolutional Neural Network (CNN) built with TensorFlow/Keras.
* The trained model can recognize digits (0–9) not only from the MNIST dataset but also from handwritten images stored in a folder.
## About MNIST dataset:
* The MNIST database (Modified National Institute of Standards and Technology database) of handwritten digits consists of a training set of 60,000 examples, and a test set of 10,000 examples.
* It is a subset of a larger set available from NIST. Additionally, the black and white images from NIST were size-normalized and centered to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.
## Features 📌
* Trains a CNN model on the MNIST dataset with >97% accuracy.
* Saves the trained model for later predictions.
* Reads handwritten digit images (.jpg/.png) from a folder and predicts them.
* Preprocessing includes:
   * Grayscale conversion
   * Thresholding & inversion
   * Stroke thickening (dilation)
   * Cropping, resizing, and padding to 28×28 pixels
   * Works on Windows, macOS, or Linux.

## Project Structure 📂 
```
MSA_proj/
│── CNN.py\n          # Train the CNN model on MNIST
│── Model.py          # Predict handwritten digits from images
│──tf_cnn_model.h5    # Saved trained model (after training)
│── imagegenerate.py  # Generates images for you according to MNIST dataset
│── my_digits/        # Place your handwritten or generated digit images here
│    └── digit1.jpg
│── README.md         # Project documentation
```

## Usage 🧪
* Place your handwritten digit images (JPG/PNG) in the digits/ folder.
* Run the prediction script.
* It should give the output as given in the example below.
* Original handwritten digits:

* <img width="28" height="28" alt="5_0" src="https://github.com/user-attachments/assets/7a670822-9f6e-490e-9d5a-0ba1047aec15" />
* <img width="28" height="28" alt="6_0" src="https://github.com/user-attachments/assets/7e932378-2585-42cf-9a6d-defa6c22bab8" />
* ![WIN_20250907_17_05_03_Pro](https://github.com/user-attachments/assets/590627ea-47dd-4a89-b656-b0dd29deae76)
* ![WIN_20250907_18_35_46_Pro](https://github.com/user-attachments/assets/8184ac2f-05a8-49fd-89f6-d245575027d3)
* ![WIN_20250907_22_37_39_Pro](https://github.com/user-attachments/assets/14e5fff3-c978-4771-8876-acd39c3ba7a7)
* Predicted digits as shown in output:
 <img width="1919" height="1127" alt="image" src="https://github.com/user-attachments/assets/37f69387-3bd9-4fc5-92b0-4f40dc972c67" />
 
## Future Improvements 🚀 
* Add a Flask/Streamlit web app for live digit recognition.
* Allow drawing digits directly with mouse input
* Experiment with different CNN architectures for higher accuracy.
  
## Author 👨‍💻 
* Developed by Abhilash
* Inspired by the classic MNIST Handwritten Digit Recognition problem.
