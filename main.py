import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import numpy as np
import cv2
from ultralytics import YOLO
from angle_utils import calculate_angle
from database import get_db, FlaggedRep

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading pose model...")
pose_model = YOLO('yolov8n-pose.pt')
print("Model loaded.")

# Folder to store flagged rep images
FLAGGED_DIR = "flagged_reps"
os.makedirs(FLAGGED_DIR, exist_ok=True)
app.mount("/flagged_reps", StaticFiles(directory=FLAGGED_DIR), name="flagged_reps")

sessions = {}

def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {
            "counter": 0,
            "stage": None,
            "rep_min_knee_angle": 999,
            "rep_max_knee_forward": 0,
            "rep_elbow_start_x": None,
            "rep_max_elbow_drift": 0,
            "rep_max_shoulder_angle": 0,
        }
    return sessions[session_id]

SQUAT_DEPTH_THRESHOLD = 110
SQUAT_KNEE_FORWARD_THRESHOLD = 40
CURL_ELBOW_DRIFT_THRESHOLD = 35
ARM_RAISE_HEIGHT_THRESHOLD = 135


def highlight_point(frame, point, label):
    """Draws a red circle + label on the joint that caused the issue."""
    x, y = int(point[0]), int(point[1])
    cv2.circle(frame, (x, y), 15, (0, 0, 255), 3)
    cv2.putText(frame, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def save_flagged_rep(db: Session, session_id, exercise, rep_number, issues, frame, highlight_pt=None, highlight_label=""):
    """Saves an annotated image to disk and logs the flagged rep in the database."""
    annotated = frame.copy()
    if highlight_pt is not None:
        annotated = highlight_point(annotated, highlight_pt, highlight_label)

    filename = f"{session_id}_{exercise}_rep{rep_number}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"
    filepath = os.path.join(FLAGGED_DIR, filename)
    cv2.imwrite(filepath, annotated)

    record = FlaggedRep(
        session_id=session_id,
        exercise=exercise,
        rep_number=rep_number,
        issue="; ".join(issues),
        image_path=filepath,
    )
    db.add(record)
    db.commit()


@app.get("/")
def read_root():
    return {"message": "Physio AI API is running"}


@app.post("/analyze-frame")
async def analyze_frame(
    file: UploadFile = File(...),
    exercise: str = Form(...),
    session_id: str = Form(...),
    db: Session = Depends(get_db),
):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    session = get_session(session_id)
    feedback = ""
    tracked_angle = 0
    alert = False

    results = pose_model(frame, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].tolist()
        shoulder, elbow, wrist = kp[6], kp[8], kp[10]
        hip, knee, ankle = kp[12], kp[14], kp[16]

        points = [shoulder, elbow, wrist, hip, knee, ankle]
        if all(p != [0, 0] for p in points):
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)
            knee_angle = calculate_angle(hip, knee, ankle)

            # ============ SQUAT ============
            if exercise == "squat":
                session["rep_min_knee_angle"] = min(session["rep_min_knee_angle"], knee_angle)
                knee_forward_dist = abs(knee[0] - ankle[0])
                session["rep_max_knee_forward"] = max(session["rep_max_knee_forward"], knee_forward_dist)

                # Check rep completion FIRST, using the current stage
                if session["stage"] == "down" and knee_angle > 160:
                    session["counter"] += 1
                    session["stage"] = "up"

                    issues = []
                    if session["rep_min_knee_angle"] > SQUAT_DEPTH_THRESHOLD:
                        issues.append("Squat not deep enough")
                    if session["rep_max_knee_forward"] > SQUAT_KNEE_FORWARD_THRESHOLD:
                        issues.append("Knees going too far past toes")

                    if issues:
                        feedback = "Incorrect: " + "; ".join(issues)
                        alert = True
                        save_flagged_rep(db, session_id, exercise, session["counter"], issues,
                                          frame, highlight_pt=knee, highlight_label="Issue here")
                    else:
                        feedback = "Correct form!"

                    print(f"[squat] min_knee_angle={session['rep_min_knee_angle']:.1f} "
                          f"max_knee_forward={session['rep_max_knee_forward']:.1f}px -> {feedback}")
                    session["rep_min_knee_angle"] = 999
                    session["rep_max_knee_forward"] = 0

                # THEN update stage for next rep tracking
                elif knee_angle > 160:
                    session["stage"] = "up"
                elif knee_angle < 130 and session["stage"] == "up":
                    session["stage"] = "down"

                tracked_angle = knee_angle

            # ============ BICEP CURL ============
            elif exercise == "bicep_curl":
                if elbow_angle > 160:
                    session["stage"] = "down"
                    session["rep_elbow_start_x"] = elbow[0]

                if session["rep_elbow_start_x"] is not None:
                    drift = abs(elbow[0] - session["rep_elbow_start_x"])
                    session["rep_max_elbow_drift"] = max(session["rep_max_elbow_drift"], drift)

                if elbow_angle < 60 and session["stage"] == "down":
                    session["stage"] = "up"
                    session["counter"] += 1

                    issues = []
                    if session["rep_max_elbow_drift"] > CURL_ELBOW_DRIFT_THRESHOLD:
                        issues.append("Elbow swinging away from body")

                    if issues:
                        feedback = "Incorrect: " + "; ".join(issues)
                        alert = True
                        save_flagged_rep(db, session_id, exercise, session["counter"], issues,
                                          frame, highlight_pt=elbow, highlight_label="Issue here")
                    else:
                        feedback = "Correct form!"

                    print(f"[bicep_curl] peak_elbow_drift={session['rep_max_elbow_drift']:.1f}px -> {feedback}")
                    session["rep_max_elbow_drift"] = 0

                tracked_angle = elbow_angle

            # ============ ARM RAISE ============
            elif exercise == "arm_raise":
                session["rep_max_shoulder_angle"] = max(session["rep_max_shoulder_angle"], shoulder_angle)

                if shoulder_angle < 40:
                    session["stage"] = "down"
                if shoulder_angle > 100 and session["stage"] == "down":
                    session["stage"] = "up"
                    session["counter"] += 1

                    issues = []
                    if session["rep_max_shoulder_angle"] < ARM_RAISE_HEIGHT_THRESHOLD:
                        issues.append("Arm not raised high enough")

                    if issues:
                        feedback = "Incorrect: " + "; ".join(issues)
                        alert = True
                        save_flagged_rep(db, session_id, exercise, session["counter"], issues,
                                          frame, highlight_pt=shoulder, highlight_label="Issue here")
                    else:
                        feedback = "Correct form!"

                    print(f"[arm_raise] peak_shoulder_angle={session['rep_max_shoulder_angle']:.1f} -> {feedback}")
                    session["rep_max_shoulder_angle"] = 0

                tracked_angle = shoulder_angle

    # Extract all 17 keypoints (as plain lists) to send back for client-side drawing
    keypoints_list = []
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints_list = results[0].keypoints.xy[0].tolist()

    return {
        "exercise": exercise,
        "reps": session["counter"],
        "feedback": feedback,
        "angle": round(tracked_angle, 1),
        "alert": alert,
        "keypoints": keypoints_list,
    }


@app.get("/flagged-reps/{session_id}")
def get_flagged_reps(session_id: str, db: Session = Depends(get_db)):
    """Returns all flagged reps for a given session, so the frontend can show a history."""
    records = db.query(FlaggedRep).filter(FlaggedRep.session_id == session_id).order_by(FlaggedRep.id.desc()).all()
    return [
        {
            "id": r.id,
            "exercise": r.exercise,
            "rep_number": r.rep_number,
            "issue": r.issue,
            "image_url": f"http://127.0.0.1:8000/flagged_reps/{os.path.basename(r.image_path)}",
            "timestamp": r.timestamp,
        }
        for r in records
    ]


@app.post("/reset-session")
def reset_session(session_id: str = Form(...)):
    sessions[session_id] = {
        "counter": 0,
        "stage": None,
        "rep_min_knee_angle": 999,
        "rep_max_knee_forward": 0,
        "rep_elbow_start_x": None,
        "rep_max_elbow_drift": 0,
        "rep_max_shoulder_angle": 0,
    }
    return {"message": "Session reset"}
