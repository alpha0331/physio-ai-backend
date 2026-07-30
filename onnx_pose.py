import numpy as np
import cv2
import onnxruntime as ort

# Load the ONNX model once at import time (equivalent to YOLO('yolov8n-pose.pt') before)
session = ort.InferenceSession('yolov8n-pose.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

CONF_THRESHOLD = 0.5
INPUT_SIZE = 640


def letterbox(image, new_size=640):
    """Resizes image to a square while preserving aspect ratio, padding the rest.
    Returns the resized image plus the scale and padding used, so we can map
    detected keypoints back to the original image's coordinates afterward."""
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_h = new_size - new_h
    pad_w = new_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, left, top


def iou(box_a, box_b):
    """Standard box IoU, used to pick the best detection when several overlap."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (ya2 - ya1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0


def get_keypoints(frame):
    """
    Runs pose detection on a raw BGR frame (same input the old YOLO(frame) call took)
    and returns a list of 17 [x, y] keypoints in the ORIGINAL frame's coordinates,
    matching the shape/format the rest of the app already expects.
    Returns an empty list if no confident person detection is found.
    """
    original_h, original_w = frame.shape[:2]

    padded, scale, pad_left, pad_top = letterbox(frame, INPUT_SIZE)
    img = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, normalize
    img = img.transpose(2, 0, 1)[np.newaxis, :]  # HWC -> CHW -> add batch dim

    outputs = session.run(None, {input_name: img})[0]  # shape: (1, 56, 8400)
    predictions = outputs[0].transpose(1, 0)  # -> (8400, 56)

    # Columns: [cx, cy, w, h, conf, kp1_x, kp1_y, kp1_v, kp2_x, kp2_y, kp2_v, ...]
    confidences = predictions[:, 4]
    best_idx_pool = np.where(confidences > CONF_THRESHOLD)[0]

    if len(best_idx_pool) == 0:
        return []

    # Keep only the single highest-confidence detection (we only ever expect one person)
    best_idx = best_idx_pool[np.argmax(confidences[best_idx_pool])]
    row = predictions[best_idx]

    keypoints_raw = row[5:]  # 17 * 3 = 51 values
    keypoints = []
    for i in range(17):
        kx, ky, kv = keypoints_raw[i * 3], keypoints_raw[i * 3 + 1], keypoints_raw[i * 3 + 2]
        if kv < 0.3:  # low-confidence individual point, treat as not detected
            keypoints.append([0.0, 0.0])
            continue
        # Undo letterbox padding/scaling to map back to the original frame size
        orig_x = (kx - pad_left) / scale
        orig_y = (ky - pad_top) / scale
        keypoints.append([float(orig_x), float(orig_y)])

    return keypoints
