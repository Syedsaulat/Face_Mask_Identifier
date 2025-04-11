# Face Mask Detection - Quick Start Guide

We've created several scripts to make it easy to run the application:

## Option 1: Run with Streamlit Web Interface

```
python run_cpu.py
```

This will:
1. Install all required packages
2. Download necessary model files
3. Start the Streamlit web interface

The web interface will be available at http://localhost:8501

## Option 2: Run with OpenCV Video Window

```
python run_video.py
```

This will:
1. Install all required packages
2. Download necessary model files
3. Open a video window using your webcam for real-time detection

Press 'q' to quit the video detection.

## Troubleshooting

If you encounter any issues:

1. Make sure your webcam is properly connected
2. Try running the CPU version if you have GPU-related errors
3. Check that all required packages are installed:
   ```
   pip install -r requirements.txt
   ```
4. Ensure the model files are downloaded correctly by running:
   ```
   python setup.py
   ```

## Note

The application requires a webcam for real-time detection. If you don't have a webcam, you can use the image detection feature:

```
python detect_mask_image.py --image images/example.jpg
```

Replace `images/example.jpg` with the path to your image file.