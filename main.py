import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

# Put the input video path here
video_path = r"YOUR_INPUT_VIDEO.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

"""
To draw polygon zone for counting
    1. Access polygonzone.roboflow.com
    2. Extract one frame from the input video
    3. Draw polygonzone
    4. Copy the coordinate array (as plain list, not np.array)
"""
# region_points = [(20, 400), (1080, 400)] # line region
# region_points = [[185, 148], [300, 150], [312, 631], [196, 631]]  # rectangle region
region_points = [[207, 159], [248, 160], [245, 617], [245, 617], [187, 615], [206, 168]]   # polygon region

poly_cnt = np.array(region_points, dtype=np.int32).reshape((-1, 1, 2))
poly_min_x = min(p[0] for p in region_points) # Left edge of the polygon zone
poly_max_x = max(p[0] for p in region_points) # Right edge of the polygon zone
EXIT_MARGIN = 10
LOST_TTL    = 5 

# Video writer
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.release()
video_writer = cv2.VideoWriter("object_counting_output.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Model
model = YOLO("yolo_apple_finetune.pt")  # load a pretrained YOLO detection model

# Helpers
def draw_zone(frame):
    """
    Tint + outline the polygon zone
    """
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_cnt], (0, 255, 255))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [poly_cnt], isClosed=True, color=(0, 255, 255), thickness=2)

def inside_polygon(cx, cy):
    """
    Check whether the center of the object is in the polygon zone or not
    """
    return cv2.pointPolygonTest(poly_cnt, (int(cx), int(cy)), False) >= 0

def put_hud(frame, in_cnt, out_cnt):
    """
    HUD counters
    """
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"IN: {in_cnt}", (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"OUT: {out_cnt}", (220, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
# Tracking states
# id -> { in_done, inside, out_done, last_cx, last_seen, prev_inside, entered_from_left }
states = {}
in_count  = 0
out_count = 0
frame_idx = 0

# GUI
win = "Apple Counter (Polygon Zone)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, min(1280, w), min(720, h))

# Device selection
try:
    chosen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
except Exception:
    chosen_device = "cpu"
print(f"Using device: {chosen_device}")

# Process with YOLO tracking
t0 = time.time()
for res in model.track(
    source=video_path,
    stream=True,
    tracker="bytetrack.yaml",       # Ultralytics tracker
    persist=True,
    imgsz=768,
    conf=0.25,
    iou=0.7,
    device=chosen_device,
    verbose=False
):
    frame = res.orig_img.copy()  # BGR
    draw_zone(frame)

    # Collect detections
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([-1]*len(xyxy))
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), int)
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    else:
        xyxy, ids, clss, conf = [], [], [], []

    # Update per detection
    for (x1, y1, x2, y2), oid, c, s in zip(xyxy, ids, clss, conf):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        is_in  = inside_polygon(cx, cy)

        st = states.get(oid, {"in_done": False, "inside": False, "out_done": False,
                              "last_cx": cx, "last_seen": frame_idx, "prev_inside": False,
                              "entered_from_left": False})

        # Check if entering from left side
        if not st["in_done"] and cx < poly_min_x:
            st["entered_from_left"] = True

        # IN: first time center enters polygon
        if is_in and not st["in_done"]:
            in_count += 1
            st["in_done"] = True

        # OUT: when the tracked center exits to the right, after having been inside and entered from left
        
        if (not is_in) and st["inside"]:
           out_count += 1
           st["inside"] = False
           st["out_done"] = True

        # Update state
        st["inside"]      = is_in
        st["prev_inside"] = st.get("prev_inside", False) or is_in
        st["last_cx"]     = cx
        st["last_seen"]   = frame_idx
        states[oid]       = st

        # DRAW ONLY if inside polygon
        if is_in:
            # label text: class name + confidence
            try:
                name = model.names[int(c)]
            except Exception:
                name = "apple"
            label = f"{name} {s:.2f}"

            # bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # label bg
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 200, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # center dot
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

    # If we lost a track after it passed the right edge, still count one OUT.
    to_mark = []
    for oid, st in states.items():
        if st["in_done"] and (not st["out_done"]) and st["entered_from_left"]:
            if frame_idx - st["last_seen"] > LOST_TTL and st["last_cx"] > poly_max_x + EXIT_MARGIN:
                out_count += 1
                st["out_done"] = True
                to_mark.append(oid)
    # clean-up old tracks to keep dict small
    for oid in to_mark:
        states[oid] = st

    # HUD display
    put_hud(frame, in_count, out_count)

    video_writer.write(frame)
    cv2.imshow(win, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord(' '):
        while True:
            k2 = cv2.waitKey(0) & 0xFF
            if k2 in (27, ord('q'), ord(' ')):
                break
        if k2 in (27, ord('q')):
            break

    frame_idx += 1

video_writer.release()
cv2.destroyAllWindows()
dur = time.time() - t0
print(f"[DONE] Saved object_counting_output.mp4 | IN={in_count} OUT={out_count} | fps~{frame_idx/max(dur,1e-6):.2f}")