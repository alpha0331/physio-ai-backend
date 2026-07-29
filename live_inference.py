import cv2
import joblib
import pandas as pd
from ultralytics import YOLO
from angle_utils import calculate_angle

print("Loading models...")
pose_model = YOLO('yolov8n-pose.pt')
print("Model loaded.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Controls: press '1' = squat, '2' = bicep_curl, '3' = arm_raise, 'q' = quit")

selected_exercise = "squat"   # default
counter = 0
stage = None

# --- Squat-specific rep tracking state ---
rep_min_knee_angle = 999
rep_max_knee_forward = 0
SQUAT_DEPTH_THRESHOLD = 110
SQUAT_KNEE_FORWARD_THRESHOLD = 40

# --- Bicep curl rep tracking state ---
rep_elbow_start_x = None      # elbow x-position when rep starts (arm down)
rep_max_elbow_drift = 0       # how far elbow moved from its starting position during the rep
CURL_ELBOW_DRIFT_THRESHOLD = 35   # pixels of allowed elbow movement before flagged as "swinging"

# --- Arm raise rep tracking state ---
rep_max_shoulder_angle = 0    # highest shoulder angle reached during the rep
ARM_RAISE_HEIGHT_THRESHOLD = 85   # shoulder angle must exceed this at the top

last_feedback = ""
last_feedback_color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].tolist()
        shoulder, elbow, wrist = kp[6], kp[8], kp[10]
        hip, knee, ankle = kp[12], kp[14], kp[16]

        points = [shoulder, elbow, wrist, hip, knee, ankle]
        if all(p != [0, 0] for p in points):
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)

            # ============ SQUAT ============
            if selected_exercise == "squat":
                rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                knee_forward_dist = knee[0] - ankle[0]
                rep_max_knee_forward = max(rep_max_knee_forward, abs(knee_forward_dist))

                if knee_angle > 160:
                    stage = "up"
                if knee_angle < 130 and stage == "up":
                    stage = "down"

                if stage == "down" and knee_angle > 160:
                    counter += 1
                    stage = "up"

                    issues = []
                    if rep_min_knee_angle > SQUAT_DEPTH_THRESHOLD:
                        issues.append("Squat not deep enough")
                    if rep_max_knee_forward > SQUAT_KNEE_FORWARD_THRESHOLD:
                        issues.append("Knees going too far past toes")

                    if issues:
                        last_feedback = "Incorrect: " + "; ".join(issues)
                        last_feedback_color = (0, 0, 255)
                    else:
                        last_feedback = "Correct form!"
                        last_feedback_color = (0, 255, 0)

                    print(f"Rep {counter}: {last_feedback}")
                    rep_min_knee_angle = 999
                    rep_max_knee_forward = 0

                tracked_angle = knee_angle

            # ============ BICEP CURL ============
            elif selected_exercise == "bicep_curl":
                if elbow_angle > 160:
                    stage = "down"
                    rep_elbow_start_x = elbow[0]

                if rep_elbow_start_x is not None:
                    drift = abs(elbow[0] - rep_elbow_start_x)
                    rep_max_elbow_drift = max(rep_max_elbow_drift, drift)

                if elbow_angle < 60 and stage == "down":
                    stage = "up"
                    counter += 1

                    issues = []
                    if rep_max_elbow_drift > CURL_ELBOW_DRIFT_THRESHOLD:
                        issues.append("Elbow swinging away from body")

                    if issues:
                        last_feedback = "Incorrect: " + "; ".join(issues)
                        last_feedback_color = (0, 0, 255)
                    else:
                        last_feedback = "Correct form!"
                        last_feedback_color = (0, 255, 0)

                    print(f"Rep {counter}: {last_feedback}")
                    rep_max_elbow_drift = 0

                tracked_angle = elbow_angle

            # ============ ARM RAISE ============
            elif selected_exercise == "arm_raise":
                rep_max_shoulder_angle = max(rep_max_shoulder_angle, shoulder_angle)

                if shoulder_angle < 40:
                    stage = "down"
                if shoulder_angle > 100 and stage == "down":
                    stage = "up"
                    counter += 1

                    issues = []
                    if rep_max_shoulder_angle < ARM_RAISE_HEIGHT_THRESHOLD:
                        issues.append("Arm not raised high enough")

                    if issues:
                        last_feedback = "Incorrect: " + "; ".join(issues)
                        last_feedback_color = (0, 0, 255)
                    else:
                        last_feedback = "Correct form!"
                        last_feedback_color = (0, 255, 0)

                    print(f"Rep {counter}: {last_feedback}")
                    rep_max_shoulder_angle = 0

                tracked_angle = shoulder_angle

            cv2.putText(annotated_frame, f'Exercise: {selected_exercise}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f'Reps: {counter}', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f'Angle: {int(tracked_angle)}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(annotated_frame, last_feedback, (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_feedback_color, 2)

    cv2.imshow('Physio AI - Live', annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        selected_exercise = "squat"
        counter, stage = 0, None
        rep_min_knee_angle, rep_max_knee_forward = 999, 0
        last_feedback = ""
        print("Switched to: squat")
    elif key == ord('2'):
        selected_exercise = "bicep_curl"
        counter, stage = 0, None
        rep_elbow_start_x, rep_max_elbow_drift = None, 0
        last_feedback = ""
        print("Switched to: bicep_curl")
    elif key == ord('3'):
        selected_exercise = "arm_raise"
        counter, stage = 0, None
        rep_max_shoulder_angle = 0
        last_feedback = ""
        print("Switched to: arm_raise")

cap.release()
cv2.destroyAllWindows()
print("Stopped.")