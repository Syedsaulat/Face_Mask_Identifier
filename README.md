# Face Mask Detection

<div align="center">
  <img src="https://github.com/Syedsaulat/Face-Mask-Detection/blob/logo/Logo/facemaskdetection.ai%20%40%2051.06%25%20(CMYK_GPU%20Preview)%20%2018-02-2021%2018_33_18%20(2).png" width="200" height="200"/>
  <h4>Face Mask Detection System built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts to detect face masks in static images and real-time video streams.</h4>
</div>

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/Syedsaulat/Face-Mask-Detection/issues)
[![Forks](https://img.shields.io/github/forks/Syedsaulat/Face-Mask-Detection.svg?logo=github)](https://github.com/Syedsaulat/Face-Mask-Detection/network/members)
[![Stargazers](https://img.shields.io/github/stars/Syedsaulat/Face-Mask-Detection.svg?logo=github)](https://github.com/Syedsaulat/Face-Mask-Detection/stargazers)
[![Issues](https://img.shields.io/github/issues/Syedsaulat/Face-Mask-Detection.svg?logo=github)](https://github.com/Syedsaulat/Face-Mask-Detection/issues)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/syedsaulat/)

![Live Demo](https://github.com/Syedsaulat/Face-Mask-Detection/blob/master/Readme_images/Demo.gif)

## Motivation
Amid the COVID-19 pandemic, efficient face mask detection applications are in high demand for transportation, densely populated areas, and enterprises to ensure safety.

## Project Demo
[![YouTube Demo](https://img.youtube.com/vi/wYwW7gAYyxw/0.jpg)](https://youtu.be/wYwW7gAYyxw)

## Tech Stack
- OpenCV
- Caffe-based face detector
- Keras
- TensorFlow
- MobileNetV2

## Features
- Accurate detection without morphed masked images dataset
- Computationally efficient using MobileNetV2 architecture
- Suitable for real-time applications and embedded systems

## Installation
1. Clone the repo:
```bash
git clone https://github.com/Syedsaulat/Face-Mask-Detection.git
cd Face-Mask-Detection
```

2. Create and activate virtual environment:
```bash
virtualenv test
source test/bin/activate
```

3. Install requirements:
```bash
pip3 install -r requirements.txt
```

## Usage
Train the model:
```bash
python3 train_mask_detector.py --dataset dataset
```

Detect masks in an image:
```bash
python3 detect_mask_image.py --image images/pic1.jpeg
```

Real-time detection:
```bash
python3 detect_mask_video.py
```

## Results
Achieved 98% accuracy with TensorFlow 2.5.0

## Contact
- Email: havisyed3@gmail.com
- LinkedIn: [Syedsaulat](https://www.linkedin.com/in/syedsaulat/)

## License
MIT © [Syedsaulat](https://github.com/Syedsaulat/Face-Mask-Detection/blob/master/LICENSE)
