import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Enable GPU memory growth to prevent TF from taking all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make predictions if at least one face was detected
    if len(faces) > 0:
        # Convert to numpy array and make batch predictions
        faces = np.array(faces, dtype="float32")
        with tf.device('/GPU:0'):  # Force TensorFlow to use GPU
            preds = maskNet.predict(faces, batch_size=32)  # Increased batch size for GPU

    return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])

# Enable CUDA backend for OpenCV's DNN module
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("[INFO] Setting up CUDA backend for OpenCV DNN")
    faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
with tf.device('/GPU:0'):  # Load model on GPU
    maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize frame processing variables
frame_count = 0
skip_frames = 2  # Process every 3rd frame to reduce load
batch_size = 8   # Process faces in batches for GPU efficiency

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    
    # Only process every nth frame to reduce processing load
    frame_count += 1
    if frame_count % (skip_frames + 1) != 0:
        continue
        
    # Resize frame while maintaining aspect ratio
    frame = imutils.resize(frame, width=800)  # Higher resolution for better detection
    
    # detect faces in the frame and predict if they are wearing masks
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # Calculate and display FPS (using time-based calculation instead of CAP_PROP_FPS)
    if 'prev_time' not in locals():
        prev_time = time.time()
        fps = 0
    else:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# cleanup
cv2.destroyAllWindows()
vs.stop()
