import cv2
import math
import numpy as np
from ultralytics import solutions

# Function to rotate a point around a center
def rotate_point(x, y, cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    x_shifted = x - cx
    y_shifted = y - cy
    x_rot = x_shifted * math.cos(angle_rad) - y_shifted * math.sin(angle_rad)
    y_rot = x_shifted * math.sin(angle_rad) + y_shifted * math.cos(angle_rad)
    return (int(cx + x_rot), int(cy + y_rot))

# Original square points
region_points = [(495, 595), (535, 595), (535, 762), (495, 762)]
# Calculate center
cx, cy = (345 + 920) / 2, (380 + 485) / 2
# Rotate by x degrees
angle =4
# region_points = [rotate_point(x, y, cx, cy, angle) for (x, y) in region_points]
 
cap = cv2.VideoCapture("video/federal.mp4")
assert cap.isOpened(), "Error reading video file"
 
# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_federal_traffic_counting_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter( 
    show=True,  # display the output
    show_in=False,
    show_out=True,
    conf=0.18, 
    iou = 0.1,
    classes=[9,10],
    tracker="bytetrack.yaml",
    region=region_points,  # pass region points
    model="yolo11x-obb.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
 
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # --- Preprocessing: Inverse, Brightness/Contrast, B&W ---
    # Inverse
    # im0 = cv2.bitwise_not(im0)
    # Brightness/Contrast
    # alpha = 2.0  # Contrast control (1.0-3.0)
    # beta = 10    # Brightness control (0-100)
    # im0 = cv2.convertScaleAbs(im0, alpha=alpha, beta=beta)
    # Convert to grayscale and then to B&W
    gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    # _, im0 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels if needed
    # --- End Preprocessing ---

    results = counter(im0)
    print(results)  # access the output
    processed_frame = results.plot_im  # Check if this frame contains the drawn results
    
    # Draw the rotated region in light green
    pts = np.array(region_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(processed_frame, [pts], isClosed=True, color=(102, 255, 102), thickness=1)

    video_writer.write(processed_frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows