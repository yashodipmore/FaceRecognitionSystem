# Face Recognition with Deep Learning and HOG Algorithm

This project demonstrates a modern face recognition system using deep learning techniques and the Histogram of Oriented Gradients (HOG) algorithm. The system is capable of detecting, aligning, encoding, and recognizing faces in images with high accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Methodology](#methodology)
  - [Face Detection (HOG Algorithm)](#face-detection-hog-algorithm)
  - [Face Alignment (Affine Transformations)](#face-alignment-affine-transformations)
  - [Face Encoding (Deep Learning)](#face-encoding-deep-learning)
  - [Face Recognition (Linear SVM)](#face-recognition-linear-svm)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

## Introduction

This project implements a face recognition pipeline using several advanced techniques:
- **Face Detection** using the Histogram of Oriented Gradients (HOG) method.
- **Face Alignment** using an ensemble of regression trees for affine transformations.
- **Face Encoding** with a deep learning model to generate face embeddings.
- **Face Recognition** using a Linear SVM classifier to match faces against a database of known individuals.

## Features

- **Face Detection**: Identifies and locates faces within an image.
- **Face Alignment**: Normalizes the face position for accurate recognition.
- **Face Encoding**: Converts faces into a 128-dimensional embedding vector.
- **Face Recognition**: Matches the encoded faces with known individuals using a classifier.

## Methodology

### Face Detection (HOG Algorithm)

We use the Histogram of Oriented Gradients (HOG) method for face detection. This method computes the weighted vote orientation gradients over 16x16 pixel squares, producing a simplified representation (HOG image) that captures the basic structure of a face.

```python
import dlib

face_detector = dlib.get_frontal_face_detector()
detected_faces = face_detector(image, 1)
face_pose_predictor = dlib.shape_predictor(predictor_model)
pose_landmarks = face_pose_predictor(image, detected_face)

# Align the face
face_aligner = openface.AlignDlib(predictor_model)
aligned_face = face_aligner.align(534, image, detected_face, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
Face Recognition (Linear SVM)
Finally, we use a Linear SVM classifier to predict the identity of the detected face by comparing the face encoding to a database of known encodings.

Installation
To set up this project, you'll need to install the following dependencies:

Python 3.x
dlib
numpy
openface
scikit-learn (for SVM classifier)
Install the dependencies using pip:
pip install dlib numpy openface scikit-learn
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/face-recognition-clone.git
cd face-recognition-clone
Run the face detection script:

bash
Copy code
python detect_faces.py --image <path_to_image>
Align the detected faces:

bash
Copy code
python align_faces.py --image <path_to_image>
Encode the faces:

bash
Copy code
python encode_faces.py --image <path_to_image>
Recognize the faces using the classifier:

bash
Copy code
python recognize_faces.py --image <path_to_image>
Acknowledgements
Special thanks to Adam Geitgey for his excellent post and guidance on face recognition techniques. This project follows the pipeline outlined in his work.

Author
Yashodip More
LinkedIn
