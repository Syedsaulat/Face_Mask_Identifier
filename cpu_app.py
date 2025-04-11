import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load model paths using absolute paths
face_detector_dir = os.path.join(current_dir, "face_detector")
prototxtPath = os.path.join(face_detector_dir, "deploy.prototxt")
weightsPath = os.path.join(face_detector_dir, "res10_300x300_ssd_iter_140000.caffemodel")
model_path = os.path.join(current_dir, "mask_detector.model")

print("[INFO] Loading models from:", current_dir)

# Load models
net = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model(model_path)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize batch processing variables
        self.batch_size = 4  # Process multiple faces in parallel
        self.face_batch = []
        self.loc_batch = []

    @staticmethod
    def preprocess_frame(frame):
        return cv2.resize(frame, (300, 300))

    def transform(self, frame):
        # Convert frame to OpenCV format and preprocess
        img = frame.to_ndarray(format="bgr24")
        processed_frame = self.preprocess_frame(img)

        # Create blob and detect faces
        blob = cv2.dnn.blobFromImage(processed_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        faces = []
        locs = []
        (h, w) = img.shape[:2]

        # Process all detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = img[startY:endY, startX:endX]
                if face.size == 0:
                    continue

                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                
                faces.append(face)
                locs.append((startX, startY, endX, endY))

                # Process in batches
                if len(faces) >= self.batch_size:
                    faces_array = np.array(faces, dtype="float32")
                    preds = model.predict(faces_array, batch_size=self.batch_size)

                    # Draw results for the batch
                    for (box, pred) in zip(locs, preds):
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred

                        label = "Mask" if mask > withoutMask else "No Mask"
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                        label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                        cv2.putText(img, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

                    faces = []
                    locs = []

        # Process any remaining faces
        if faces:
            faces_array = np.array(faces, dtype="float32")
            preds = model.predict(faces_array, batch_size=len(faces))

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(img, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

        return img

def main():
    st.set_page_config(page_title="Face Mask Detection")
    st.title("Real-Time Face Mask Detection")
    st.write("Using CPU for processing")

    # Configure WebRTC
    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_processing=True
    )

if __name__ == "__main__":
    main()