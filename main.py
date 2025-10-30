import cv2
import numpy as np
import torch
from ultralytics import YOLO

video_path = r"/Users/tit/Documents/deeplearning/Screen Recording 2025-10-06 221557.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# region_points = [(20, 400), (1080, 400)] # line region
# region_points = [[185, 148], [300, 150], [312, 631], [196, 631]]  # rectangle region
region_points = [[120, 174], [693, 220], [688, 267], [688, 267], [114, 220]]   # polygon region
poly_cnt = np.array(region_points, dtype=np.int32).reshape((-1, 1, 2))
poly_max_x = max(p[0] for p in region_points)
EXIT_MARGIN = 6  # đối tượng vượt quá cạnh phải của polygon + margin => OUT

# Video writer
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.release()
video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Model
model = YOLO("yolo_apple_finetune.pt")  # đổi sang đường dẫn checkpoint của bạn

# GUI
win_name = "Apple Counter (Polygon Zone)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, min(1280, w), min(720, h))

# Tracking states & counters
states = {}  # id -> {"inside": False, "in_done": False, "out_done": False}
in_count = 0
out_count = 0

# Helpers
def point_in_poly(cx, cy):
    return cv2.pointPolygonTest(poly_cnt, (int(cx), int(cy)), False) >= 0

def draw_zone(frame):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly_cnt], (0, 255, 255))
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
    cv2.polylines(frame, [poly_cnt], True, (0, 255, 255), 2)

# Device selection
try:
    chosen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
except Exception:
    chosen_device = "cpu"
print(f"Using device: {chosen_device}")

# Process with YOLO tracking
for res in model.track(
    source=video_path,
    stream=True,
    tracker="bytetrack.yaml",
    persist=True,
    imgsz=768,
    conf=0.25,
    iou=0.7,
    device=chosen_device,
    classes=[0]                 # If dataset has many classes and "apple" is class 0; remove if only 1 class
):
    frame = res.orig_img.copy()

    # Draw polygon zone
    draw_zone(frame)

    # Tracking
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)        # [N,4]
        ids  = res.boxes.id
        ids  = ids.cpu().numpy().astype(int) if ids is not None else np.array([None]*len(xyxy))

        for (x1, y1, x2, y2), oid in zip(xyxy, ids):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            inside = point_in_poly(cx, cy)

            # Init state
            st = states.get(oid, {"inside": False, "in_done": False, "out_done": False})

            # 1. Increase IN if object enters the zone
            if inside and not st["in_done"]:
                in_count += 1
                st["in_done"] = True
                st["inside"] = True

            # 2. Object is already inside the zone
            elif inside and st["in_done"]:
                st["inside"] = True

            # 3. Increase OUT if object exits the zone
            elif (not inside) and st["inside"] and (cx > poly_max_x + EXIT_MARGIN) and (not st["out_done"]):
                out_count += 1
                st["inside"] = False
                st["out_done"] = True
            else:
                st["inside"] = inside

            states[oid] = st

            # Draw bounding boxes if object inside polygon zone
            if inside:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                if oid is not None:
                    cv2.putText(frame, f"ID {oid}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                # bbox center
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

    # HUD
    cv2.putText(frame, f"IN: {in_count}", (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"OUT: {out_count}", (180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Save + show
    video_writer.write(frame)
    cv2.imshow(win_name, frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):  # Press 'ESC' or 'q' to quit
        break
    if k == ord(' '):        # Space: pause/resume
        while True:
            k2 = cv2.waitKey(0) & 0xFF
            if k2 in (ord(' '), 27, ord('q')):
                break
        if k2 in (27, ord('q')):
            break

video_writer.release()
cv2.destroyAllWindows()