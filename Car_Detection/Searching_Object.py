import os
import cv2
import torch
from ultralytics import YOLO

# 원하는 모델 넣기
model = YOLO("model")

# 원본 이미지랑 저장 위치 설정
input_folder = "path"
output_folder = "path"
os.makedirs(output_folder, exist_ok=True)

# 이미지 처리
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image = cv2.imread(image_path)

    results = model(image)
    detections = results[0].boxes

    if not detections:
        print(f"No objects found in {image_name}")
        continue

    # 인식된 객체 중에서서 가장 큰 객체 찾기 (너비 x 높이 기준)
    largest_box = max(
        detections,
        key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
    )

    x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0])

    cropped_object = image[y_min:y_max, x_min:x_max]

    # 저장
    output_path = os.path.join(output_folder, f"{image_name}")
    cv2.imwrite(output_path, cropped_object)
    print(f"Saved largest object to {output_path}")
