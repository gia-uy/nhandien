import cv2
import numpy as np
import json
from ultralytics import YOLO

VIDEO_SOURCE = 0
WINDOW_NAME = "Security Monitoring"

CONF_THRESHOLD = 0.75
PERSON_CLASS_ID = 0         # class person trong model của bạn

IMG_SIZE = 224            

FLIP_CAMERA = True          # lật 
DETECT_EVERY_N_FRAMES = 5
polygon_points = []
roi_locked = False

# LOAD MODEL

model = YOLO("nhandien/Data/best.pt")

# MOUSE EVENT
def mouse_callback(event, x, y, flags, param):
    global polygon_points, roi_locked

    if roi_locked:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"Added point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(polygon_points) > 0:
            removed_point = polygon_points.pop()
            print(f"Removed point: {removed_point}")


# ROI FUNCTIONS
def draw_roi(frame):
    for point in polygon_points:
        cv2.circle(frame, point, 5, (0, 255, 255), -1)

    if len(polygon_points) >= 2:
        for i in range(len(polygon_points) - 1):
            cv2.line(frame, polygon_points[i], polygon_points[i + 1], (0, 255, 255), 2)

    if roi_locked and len(polygon_points) >= 3:
        pts = np.array(polygon_points, dtype=np.int32)

        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 255, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def is_point_inside_polygon(point, polygon):
    pts = np.array(polygon, dtype=np.int32)

    result = cv2.pointPolygonTest(
        pts,
        point,
        False
    )

    return result >= 0


def save_roi_to_json(filename="roi_points.json"):
    data = {
        "polygon_points": polygon_points
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Saved ROI to {filename}")

# YOLO DETECTION LOGIC
def run_yolo_detection(detect_frame):
    intrusion_detected = False
    detected_objects = []

    results = model(
        detect_frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id != PERSON_CLASS_ID:
            continue

        if confidence < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center_point = (center_x, center_y)

        inside_roi = is_point_inside_polygon(center_point, polygon_points)

        if inside_roi:
            intrusion_detected = True
            status = "INTRUSION"
        else:
            status = "PERSON"

        detected_objects.append({
            "bbox": (x1, y1, x2, y2),
            "center": center_point,
            "confidence": confidence,
            "inside_roi": inside_roi,
            "status": status
        })

    return intrusion_detected, detected_objects
def draw_detections(display_frame, detected_objects):
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        center_point = obj["center"]
        confidence = obj["confidence"]
        inside_roi = obj["inside_roi"]
        status = obj["status"]

        if inside_roi:
            box_color = (0, 0, 255)
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        cv2.circle(
            display_frame,
            center_point,
            5,
            box_color,
            -1
        )

        cv2.putText(
            display_frame,
            f"{status} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2
        )

# MAIN PROGRAM

cap = cv2.VideoCapture(VIDEO_SOURCE)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

if not cap.isOpened():
    print("Không mở được camera/video.")
    exit()

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("HƯỚNG DẪN:")
print("- Click chuột trái: thêm điểm ROI")
print("- Click chuột phải: xóa điểm cuối")
print("- Nhấn S: khóa ROI")
print("- Nhấn R: vẽ lại ROI")
print("- Nhấn P: in tọa độ ROI")
print("- Nhấn Q: thoát chương trình")

frame_count = 0
last_detections = []
last_intrusion_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được frame từ camera/video.")
        break

    frame_count += 1

    if FLIP_CAMERA:
        frame = cv2.flip(frame, 1)

    # Frame gốc để YOLO detect
    detect_frame = frame.copy()

    # Frame hiển thị để vẽ ROI, bbox, cảnh báo
    display_frame = frame.copy()

    # Vẽ ROI lên frame hiển thị
    draw_roi(display_frame)

    intrusion_detected = False


    if roi_locked and len(polygon_points) >= 3:

        # Chỉ detect mỗi N frame để giảm giật
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            last_intrusion_detected, last_detections = run_yolo_detection(detect_frame)

        # Vẽ lại kết quả detect gần nhất
        draw_detections(display_frame, last_detections)

        intrusion_detected = last_intrusion_detected

    # Hiện cảnh báo nếu có xâm nhập
    if intrusion_detected:
        cv2.putText(
            display_frame,
            "CANH BAO XAM NHAP!",
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Hướng dẫn trên màn hình
    cv2.putText(
        display_frame,
        "S: lock | R: reset | P: print | Q: quit",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    if roi_locked:
        cv2.putText(
            display_frame,
            "ROI LOCKED",
            (10, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )
    else:
        cv2.putText(
            display_frame,
            "DRAW ROI",
            (10, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1
        )

    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        if len(polygon_points) >= 3:
            roi_locked = True
            save_roi_to_json()
            print("Đã khóa vùng ROI.")
            print("Tọa độ ROI:", polygon_points)
        else:
            print("Cần ít nhất 3 điểm để tạo polygon ROI.")

    elif key == ord("r"):
        polygon_points = []
        roi_locked = False
        last_detections = []
        last_intrusion_detected = False
        print("Đã reset ROI.")

    elif key == ord("p"):
        print("Tọa độ ROI hiện tại:")
        print(polygon_points)

cap.release()
cv2.destroyAllWindows()