This project implements real-time object detection and tracking for vehicles using YOLOv3 and Kalman filtering. It provides functionalities like:

- Vehicle Detection: Employs YOLOv3 to identify vehicles (class ID 2) within video frames.
- Kalman Filter Tracking: Predicts vehicle positions across frames for smoother tracking and velocity estimation.
- Hungarian Algorithm: Efficiently associates detections with tracked objects.
- Speed Calculation: Estimates vehicle speed in meters per second based on calibrated pixel-to-meter conversion.

# Features:

- Accurate object detection using YOLOv3.
- Smooth and robust vehicle tracking with Kalman filtering.
- Efficient association of detections with tracked objects.
- Real-time performance for video stream processing.

# Requirements:

- Python 3.x
- OpenCV (cv2)
- NumPy (np)
- SciPy (scipy) (for linear_sum_assignment)
- Matplotlib (optional, for visualization)

# Installation:

- Ensure you have Python 3.x installed.
- Install the required libraries using pip:
    ```pip install opencv-python numpy scipy matplotlib```


# Usage:
- Place your pre-trained YOLOv3 weights (yolov3.weights), configuration (yolov3.cfg), and class names file (coco.names) in the project directory.
- Update the pixel_to_meter function in the code with your calibrated value based on camera settings.
- Replace 'video.mp4' in the cap = cv2.VideoCapture('video.mp4') line with the path to your video file.
- Run the script:
    ```python rapid_recognition.py```

# Disclaimer:

This project is provided for educational and research purposes. Vehicle detection and tracking accuracy may vary depending on video quality, lighting conditions, and scene complexity. Calibrate the pixel_to_meter conversion for reliable speed estimation.

# License:

This project is licensed under the [MIT License](License.txt).

# Further Enhancements:
- Explore alternative object detection models for improved accuracy or efficiency.
- Implement multi-object tracking for handling multiple vehicles simultaneously.
- Integrate with display libraries for real-time visualization of bounding boxes and speed information.
- Consider incorporating deep learning-based speed estimation methods for potentially higher accuracy.

# Contributing:

We welcome contributions to this project! Feel free to submit pull requests with enhancements, bug fixes, or improvements to the code or documentation.

# Contact:

Please feel free to reach out for any questions or feedback.

# Additional Notes:
- Feel free to adjust the README.md content to better reflect your specific project goals and features.    Consider adding examples, visualizations, or code snippets for improved clarity.
- Ensure the installation instructions are accurate and up-to-date with the required libraries.
