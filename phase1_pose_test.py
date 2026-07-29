import cv2
from ultralytics import YOLO
from angle_utils import calculate_angle

print("Loading model...")
model = YOLO('yolov8n-pose.pt')
print("Model loaded.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Running... look for the window. Press 'q' in the window to quit.")

counter = 0
stage = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0]

        shoulder = keypoints[6].tolist()
        elbow = keypoints[8].tolist()
        wrist = keypoints[10].tolist()

        if shoulder != [0,0] and elbow != [0,0] and wrist != [0,0]:
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 160:
                stage = "down"
            if angle < 60 and stage == "down":
                stage = "up"
                counter += 1

            # Draw text on the video frame itself
            cv2.putText(annotated_frame, f'Angle: {int(angle)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f'Reps: {counter}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Physio AI - Live', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")