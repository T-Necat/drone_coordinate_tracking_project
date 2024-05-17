import cv2
import numpy as np


def calculate_center_shift(p0, p1, st):
    good_new = p1[st == 1]  # Selects points that are successfully tracked in the second frame.
    good_old = p0[st == 1]  # Selects points that are successfully tracked in the first frame.

    shift_x = np.mean(good_new[:, 0] - good_old[:, 0])  # Calculates the average shift in the X direction.
    shift_y = np.mean(good_new[:, 1] - good_old[:, 1])  # Calculates the average shift in the Y direction.

    return shift_x, shift_y  # Returns the shifts in the X and Y directions.


def calculate_pixel_to_meter_ratio(altitude, fov, image_width=1920, image_height=1080):
    # Convert FOV to radians
    fov_rad = np.deg2rad(fov)

    # Calculate the width of the image area (altitude / tan(FOV/2))

    # Calculates the width of the image area at ground level.
    image_width_meters = 2 * altitude * np.tan(fov_rad / 2)  # Calculates the tangent of half of the FOV.

    # Calculates the height at ground level using the ratio below.
    image_height_meters = image_width_meters * (
                image_height / image_width)  # image_height / image_width: Calculates the aspect ratio of the image.

    # Calculate the pixel-to-meter ratio
    pixel_to_meter_x = image_width_meters / image_width
    pixel_to_meter_y = image_height_meters / image_height

    return pixel_to_meter_x, pixel_to_meter_y


# Open the video source
cap = cv2.VideoCapture('/Users/tng/Code/Merküt_Code/merküt_model_test/2021 Örnek Video kopyası.mp4')

# ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Initial coordinates
x0, y0 = 0.0, 0.0
x_shift_total, y_shift_total = 0.0, 0.0

# Pixel to meter conversion ratio (for example) if we take this as input, we can run the function for any situation
altitude = 50  # Altitude of the aircraft (meters)
fov = 90  # Field of view (degrees)
pixel_to_meter_x, pixel_to_meter_y = calculate_pixel_to_meter_ratio(altitude, fov)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    shift_x, shift_y = calculate_center_shift(p0, p1, st)
    x_shift_total += shift_x
    y_shift_total += shift_y

    # Change in meters
    x_shift_meter = shift_x * pixel_to_meter_x
    y_shift_meter = shift_y * pixel_to_meter_y

    # Total position
    x_total_meter = x0 + x_shift_total * pixel_to_meter_x
    y_total_meter = y0 + y_shift_total * pixel_to_meter_y

    print(
        f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} - Shift X: {x_shift_meter:.2f} meters, Shift Y: {y_shift_meter:.2f} meters")
    print(f"Total Position - X: {x_total_meter:.2f} meters, Y: {y_total_meter:.2f} meters\n")

    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cap.release()
cv2.destroyAllWindows()