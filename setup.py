import os
import sys
import urllib.request
import zipfile
import shutil

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download complete!")

def main():
    # Create face_detector directory if it doesn't exist
    if not os.path.exists("face_detector"):
        os.makedirs("face_detector")
        print("Created face_detector directory")

    # Download face detector model files
    face_detector_files = [
        ("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", 
         "face_detector/deploy.prototxt"),
        ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", 
         "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    ]

    for url, path in face_detector_files:
        if not os.path.exists(path):
            download_file(url, path)
        else:
            print(f"{path} already exists, skipping download")

    # Download pre-trained mask detector model
    mask_model_url = "https://github.com/Syedsaulat/Face-Mask-Detection/raw/master/mask_detector.model"
    mask_model_path = "mask_detector.model"
    
    if not os.path.exists(mask_model_path):
        download_file(mask_model_url, mask_model_path)
    else:
        print(f"{mask_model_path} already exists, skipping download")

    print("Setup complete! You can now run the application.")

if __name__ == "__main__":
    main()