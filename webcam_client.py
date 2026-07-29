import cv2
import requests

API_URL = "http://127.0.0.1:8000/analyze-frame"
SESSION_ID = "webcam_test_session"

# COCO 17-keypoint skeleton connections (pairs of keypoint indices to draw lines between)
SKELETON_CONNECTIONS = [
    (5, 6),   # left shoulder - right shoulder
    (5, 7), (7, 9),      # left arm: shoulder-elbow-wrist
    (6, 8), (8, 10),     # right arm: shoulder-elbow-wrist
    (5, 11), (6, 12),    # shoulders to hips
    (11, 12),            # left hip - right hip
    (11, 13), (13, 15),  # left leg: hip-knee-ankle
    (12, 14), (14, 16),  # right leg: hip-knee-ankle
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Controls: '1' = squat, '2' = bicep_curl, '3' = arm_raise, 'r' = reset session, 'q' = quit")

selected_exercise = "squat"
last_result = {"reps": 0, "feedback": "", "angle": 0, "alert": False, "keypoints": []}

FRAME_SKIP = 3
frame_count = 0

def reset_session():
    try:
        requests.post("http://127.0.0.1:8000/reset-session", data={"session_id": SESSION_ID}, timeout=2)
        print("Session reset.")
    except requests.exceptions.RequestException as e:
        print(f"Could not reset session: {e}")

def draw_skeleton(frame, keypoints):
    """Draws circles on each keypoint and lines connecting them, matching YOLO-Pose's own style."""
    if not keypoints or len(keypoints) < 17:
        return frame

    # Draw joints
    for x, y in keypoints:
        if x == 0 and y == 0:
            continue  # not detected
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Draw connecting lines
    for a, b in SKELETON_CONNECTIONS:
        xa, ya = keypoints[a]
        xb, yb = keypoints[b]
        if (xa == 0 and ya == 0) or (xb == 0 and yb == 0):
            continue  # skip if either point wasn't detected
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 165, 0), 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        success, encoded_image = cv2.imencode('.jpg', frame)
        if success:
            files = {'file': ('frame.jpg', encoded_image.tobytes(), 'image/jpeg')}
            data = {'exercise': selected_exercise, 'session_id': SESSION_ID}

            try:
                response = requests.post(API_URL, files=files, data=data, timeout=2)
                if response.status_code == 200:
                    last_result = response.json()
                else:
                    print(f"API error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

    # Draw skeleton using the latest known keypoints
    frame = draw_skeleton(frame, last_result.get("keypoints", []))

    # Draw text overlay
    color = (0, 0, 255) if last_result.get("alert") else (0, 255, 0)
    cv2.putText(frame, f'Exercise: {selected_exercise}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Reps: {last_result.get("reps", 0)}', (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Angle: {last_result.get("angle", 0)}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if last_result.get("feedback"):
        cv2.putText(frame, last_result["feedback"], (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if last_result.get("alert"):
        cv2.putText(frame, "!!! FORM ALERT !!!", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow('Physio AI - Webcam to API', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        selected_exercise = "squat"
        reset_session()
        print("Switched to: squat")
    elif key == ord('2'):
        selected_exercise = "bicep_curl"
        reset_session()
        print("Switched to: bicep_curl")
    elif key == ord('3'):
        selected_exercise = "arm_raise"
        reset_session()
        print("Switched to: arm_raise")
    elif key == ord('r'):
        reset_session()

cap.release()
cv2.destroyAllWindows()
print("Stopped.")
