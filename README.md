# Object Tracking with Kalman Filter

## Overview
This project demonstrates real-time object tracking using OpenCV and a Kalman Filter. The system captures video from a webcam, applies background subtraction to detect moving objects, and then uses a Kalman Filter to predict the object's future position. The detected object is highlighted with a bounding box, and its predicted position is marked with a red dot.

## Features
- Real-time video capture using OpenCV
- Background subtraction for object detection
- Morphological operations to reduce noise
- Contour detection to identify the largest moving object
- Kalman Filter for motion prediction

## Requirements
To run this project, you need to have the following installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)

You can install the dependencies using:
```bash
pip install opencv-python numpy
```

## How It Works
1. **Capture Video**: The script opens the webcam and continuously reads frames.
2. **Background Subtraction**: The algorithm applies `cv2.createBackgroundSubtractorMOG2` to extract moving objects.
3. **Noise Reduction**: Morphological operations (`cv2.morphologyEx`) help clean the mask.
4. **Contour Detection**: The largest moving object is identified.
5. **Bounding Box & Measurement**: A rectangle is drawn around the detected object.
6. **Kalman Filter Prediction**:
   - The object's center is measured.
   - The Kalman Filter updates its state using the measurement.
   - A red dot indicates the predicted future position.
7. **Display & Exit**: The video feed is displayed, and pressing `q` exits the program.

## Usage
Run the script with:
```bash
python object_tracking.py
```
To stop the program, press `q` while the video window is open.

## Example Code
Here is the core implementation of the tracking system:
```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            current_measurement = np.array([[np.float32(x + w // 2)], [np.float32(y + h // 2)]])
            kalman.correct(current_measurement)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    current_prediction = kalman.predict()
    predict_x, predict_y = int(current_prediction[0]), int(current_prediction[1])
    cv2.circle(frame, (predict_x, predict_y), 5, (0, 0, 255), -1)
    cv2.imshow("Object Tracking with Kalman Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Output
- **Green bounding box**: Detected object
- **Red dot**: Kalman Filter prediction

## Future Improvements
- Improve object detection for multiple objects
- Implement different tracking algorithms for comparison
- Enhance Kalman Filter tuning for better predictions



Happy coding! 🚀

