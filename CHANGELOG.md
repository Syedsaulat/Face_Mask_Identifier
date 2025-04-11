# Changelog

## [2.0.0] - 2025-04-11
### Updated
- Upgraded TensorFlow to version 2.15.0
- Updated OpenCV requirements to version 4.8.0 or higher
- Updated NumPy requirement to 1.23.5 or higher
- Updated Matplotlib to version 3.4.1 or higher
- Updated Streamlit to version 0.79.0 or higher
- Updated Streamlit-WebRTC to version 0.45.0 or higher
- Updated Pillow requirement to version 8.3.2 or higher
- Updated SciPy to version 1.6.2 or higher
- Updated scikit-learn to version 0.24.1 or higher

### Added
- Support for Python 3.11+
- Improved GPU support with TensorFlow 2.15.0
- Enhanced real-time detection performance
- Web interface improvements with latest Streamlit

### Fixed
- Compatibility issues with newer Python versions
- Performance optimizations for CPU and GPU modes
- WebRTC stream stability improvements

## [1.1.0] - 2023-11-15

### Fixed
- Replaced problematic FPS calculation in detect_mask_video.py
- Fixed 'WebcamVideoStream' object has no attribute 'get' error
- Improved error handling for video stream operations

### Changed
- Implemented more accurate time-based FPS calculation
- Updated video processing to handle edge cases better

### Added
- Better GPU utilization documentation
- Time-based performance metrics
