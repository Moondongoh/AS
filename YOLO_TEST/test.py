# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO('yolov8n.pt')

# # Open the webcam (use 0 for the default camera)
# cap = cv2.VideoCapture(1)

# # Loop through the webcam frames
# while cap.isOpened():
#     # Read a frame from the webcam
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Webcam Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if there's an issue reading the frame
#         break

# # Release the webcam and close the display window
# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the image (test.png)
image = cv2.imread('D:/Git_Hub/AS/YOLO_TEST/t.jpg')

# Check if the image was loaded successfully
if image is not None:
    # Run YOLOv8 inference on the image
    results = model(image)

    # Visualize the results on the image
    annotated_image = results[0].plot()

    # Display the annotated image
    cv2.imshow("YOLOv8 Image Inference", annotated_image)

    # Wait until a key is pressed to close the window
    cv2.waitKey(0)

    # Close the display window
    cv2.destroyAllWindows()
else:
    print("Error: Could not load image 'test.png'")
