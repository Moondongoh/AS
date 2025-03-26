import cv2
import numpy as np
from ultralytics import YOLO

# 모델 경로 및 클래스 정의
model_path = "D:/New_Tracking/runs/detect/train5/weights/best.pt"
model = YOLO(model_path)

CLASSES = ['K3', 'Sorento', 'Forte', 'TheNewGrandStarex', 'G90']

# 전역 변수
current_frame = None
detections = []

def on_mouse_click(event, x, y, flags, param):
    global current_frame, detections

    if event == cv2.EVENT_LBUTTONDOWN:
        print("\nMouse clicked. Detecting objects...")
        
        # 클릭할 때 객체 탐지 실행
        detections = detect_objects(current_frame)

        if not detections:
            print("No objects detected.")
        else:
            print(f"Detected {len(detections)} object(s):")
            for i, detection in enumerate(detections, start=1):
                bbox = detection['bbox']
                class_id = detection['class_id']
                score = detection['score']
                class_name = CLASSES[class_id]
                
                print(f"  Object {i}:")
                print(f"    - Class: {class_name}")
                print(f"    - BBox: {bbox}")
                print(f"    - Confidence: {score:.2f}")

        # 탐지 결과 표시
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            class_name = CLASSES[class_id]
            cv2.putText(
                current_frame,
                class_name,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            cv2.rectangle(
                current_frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )
        # 클릭 후 업데이트된 프레임 표시
        cv2.imshow("Video", current_frame)

def detect_objects(frame):
    results = model.predict(frame, verbose=False)
    
    detections = []
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.5:
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': int(class_id),
                'score': score
            })
    return detections

# 영상 파일 경로
video_path = "C:/Users/as/Desktop/Forte.mp4"

# G90
# TheNewGrandStarex
# K3
# Forte
# Sorento
# TestSet


cap = cv2.VideoCapture(video_path)

# 프레임 속도 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # 기본값 설정 (비정상적인 경우 대비)

# 0.25배 속도로 재생: 기본 프레임 대기 시간의 4배
frame_delay = int((1000 / fps) * 2)

window_name = "Video"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)
cv2.setMouseCallback(window_name, on_mouse_click)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임 저장
    current_frame = frame.copy()

    # 현재 프레임 표시 (객체 탐지 없이 원본 프레임만)
    cv2.imshow(window_name, current_frame)

    # ESC 키를 눌러 종료
    key = cv2.waitKey(frame_delay) & 0xFF
    if key == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()