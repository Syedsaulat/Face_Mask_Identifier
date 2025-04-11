import os
import sys
import subprocess
import platform

def install_requirements():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Package installation complete!")

def setup_models():
    print("Setting up model files...")
    subprocess.check_call([sys.executable, "setup.py"])
    print("Model setup complete!")

def run_app():
    print("Starting the Face Mask Detection application...")
    subprocess.check_call([sys.executable, "app.py"])

def main():
    # Check if requirements are installed
    try:
        import tensorflow
        import cv2
        import streamlit
        print("Required packages already installed.")
    except ImportError:
        install_requirements()
    
    # Setup model files
    setup_models()
    
    # Run the application
    run_app()

if __name__ == "__main__":
    main()