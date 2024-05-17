# Drone Tracking Project

This project uses computer vision techniques to track the movement of a drone. It calculates the shift in X and Y coordinates based on video frames and converts these shifts to real-world distances.

## Features

- Calculates the center shift between consecutive frames using optical flow.
- Converts pixel shifts to real-world distances using the altitude and field of view of the camera.
- Outputs the total position shift in meters.

## Requirements

- OpenCV
- NumPy

## How to Run

1. Ensure you have the required libraries installed:
    ```bash
    pip install opencv-python numpy
    ```

2. Run the script with a video file as input:
    ```bash
    python drone_tracking.py
    ```

## Code Overview

- **calculate_center_shift(p0, p1, st):** Computes the average shift in X and Y directions based on tracked points between two frames.
- **calculate_pixel_to_meter_ratio(altitude, fov, image_width=1920, image_height=1080):** Calculates the conversion ratio from pixels to meters.
- **Main Script:** Reads a video file, computes the optical flow, and tracks the movement of the drone, printing the shift in meters for each frame.

## Example Output

The script prints the shift in meters for each frame and the total position in meters.

Frame 1 - Shift X: 0.05 meters, Shift Y: 0.03 meters. 
Total Position - X: 0.05 meters, Y: 0.03 meters

## License

This project is licensed under the MIT License.