import cv2
import numpy as np
from ultralytics import YOLO

#model_path = "d:/Tracking/runs/detect/train39/weights/best.pt"
model_path = "D:/New_Tracking/runs/detect/train5/weights/best.pt"
model = YOLO(model_path)

CLASSES =  ['K3', 'Sorento','Forte', 
        'TheNewGrandStarex','G90']
# ['K3', 'K5', 'K7', 'K9', 
#         'G80_Sport', 'NiroHybrid', 'Stinger', 'Spotage',
#         'Sorento', 'Forte', 'G90', 'i30', 
#         'KONA 1.6T', 'QM6', 'The_New_Grand_Starex',
#         'Maxcruz', 'Starex', 'Palisade']
def on_mouse_click(event, x, y, flags, param):
    global image, detections

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = np.array([x, y])
        
        closest_object = None
        min_distance = float('inf')

        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            distance = np.linalg.norm(clicked_point - center)
            
            if distance < min_distance:
                min_distance = distance
                closest_object = detection
        
        if closest_object:
            class_name = CLASSES[closest_object['class_id']]
            print(f"Clicked on: {class_name}")

            bbox = closest_object['bbox']
            cv2.putText(
                image,
                class_name,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )
            cv2.imshow("Image", image)

image_path = "D:/train/test/G90_3845.jpg"
image = cv2.imread(image_path)

#K3_5896
#Sorento_2688
#Forte_13228
#TheNewGrandStarex_1298
#G90_3845

def detect_objects(image):
    results = model.predict(image, verbose=False)
    
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

detections = detect_objects(image)

window_name = "Image"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
cv2.resizeWindow(window_name, 800, 600)         

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", on_mouse_click)
# Add an infinite loop to wait for ESC key
while True:
    key = cv2.waitKey(1)  # Wait for 1 ms
    if key == 27:  # ESC key has ASCII value 27
        break
cv2.destroyAllWindows()
