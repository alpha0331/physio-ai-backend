import cv2
import csv
import os
from ultralytics import YOLO
from angle_utils import calculate_angle

# CHANGE THIS before each recording session
EXERCISE_LABEL = "arm_raise"   # e.g. "squat", "arm_raise", "bicep_curl"

CSV_FILE = "exercise_data.csv"

print("Loading model...")
model = YOLO('yolov8n-pose.pt')
print("Model loaded.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# Create CSV with headers if it doesn't exist yet
file_exists = os.path.isfile(CSV_FILE)
csv_file = open(CSV_FILE, mode='a', newline='')
writer = csv.writer(csv_file)
if not file_exists:
    writer.writerow(['elbow_angle', 'shoulder_angle', 'hip_angle', 'knee_angle', 'label'])

print(f"Recording data for label: '{EXERCISE_LABEL}'. Perform the exercise now.")
print("Press 'q' to stop recording.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].tolist()

        # Right side joints: shoulder(6) elbow(8) wrist(10) hip(12) knee(14) ankle(16)
        shoulder, elbow, wrist = kp[6], kp[8], kp[10]
        hip, knee, ankle = kp[12], kp[14], kp[16]

        # Only record if all needed points are detected
        points = [shoulder, elbow, wrist, hip, knee, ankle]
        if all(p != [0, 0] for p in points):
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)

            writer.writerow([elbow_angle, shoulder_angle, hip_angle, knee_angle, EXERCISE_LABEL])

            cv2.putText(annotated_frame, f'Recording: {EXERCISE_LABEL}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Data Collection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"Done. Data saved to {CSV_FILE}")