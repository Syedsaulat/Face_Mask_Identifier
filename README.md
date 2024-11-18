Face Mask Detection Using Deep Learning
Project Description
This project aims to develop a real-time face mask detection system using deep learning techniques. The system leverages OpenCV for video stream processing and TensorFlow/Keras for mask detection. It detects faces in real-time and classifies them as either "Mask" or "No Mask."

The project uses a pre-trained deep learning model and a Haar Cascade Classifier for face detection, providing real-time feedback on the mask status.

Key Features
Real-time face detection using Haar Cascade Classifier.
Mask detection powered by a pre-trained deep learning model (MobileNetV2).
Live video stream displaying mask status (Green for Mask, Red for No Mask).
Easy setup and simple interface.
Technologies Used
Python - The main programming language.
TensorFlow & Keras - Used for the mask detection model.
OpenCV - For capturing video and performing face detection.
Numpy & Matplotlib - For data handling and visualization.
Setup Instructions
Clone the repository:
git clone https://github.com/Syedsaulat/Face_Mask_Identifier.git
Install the dependencies: Create and activate a virtual environment (optional but recommended):
The system will start the video stream and detect whether the person in front of the camera is wearing a mask or not. If a mask is detected, the system will display "Mask" in green; if no mask is detected, it will display "No Mask" in red.
