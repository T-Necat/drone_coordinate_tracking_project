# 🛸 Drone Coordinate Tracking Project
This project tracks the movement of a drone using computer vision techniques. It calculates shifts in X and Y coordinates and converts them to real-world distances.

## Features
- Uses Optical Flow to compute center shifts between consecutive frames.
- Converts pixel shifts to real-world distances based on camera altitude and field of view.
- Calculates total position shifts in meters for each frame.

## Technologies Used
- Python
- YOLO (You Only Look Once)
- OpenCV
- NumPy
- Pandas

## Requirements
Ensure the following libraries are installed:
```bash
pip install opencv-python numpy pandas
```

## How to Run
Run the following command with a video file as input:
```bash
python drone_tracking.py
```

## Code Overview
- **calculate_center_shift(p0, p1, st):** Computes the average shift in X and Y directions based on optical flow between two frames.
- **calculate_pixel_to_meter_ratio(altitude, fov, image_width=1920, image_height=1080):** Calculates the conversion ratio from pixels to meters.
- **Main Script:** Reads a video file, computes optical flow, and tracks the drone's movement, printing the shift in meters for each frame.

## Example Output
```
Frame 1 - Shift X: 0.05 meters, Shift Y: 0.03 meters. Total Position - X: 0.05 meters, Y: 0.03 meters
```

## License
This project is licensed under the MIT License.

---
Contact: [t.necatgok@gmail.com](mailto:t.necatgok@gmail.com)

---
