import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture('tcpclientsrc host=192.168.1.245 port=7001 ! decodebin ! queue ! videoconvert ! videoscale ! video/x-raw,width=1280,height=720 ! appsink drop=1', cv2.CAP_GSTREAMER)

while cap.isOpened():
    success, frame = cap.read()
    k = cv2.waitKey(1)

    if k != -1:
        break
    if success:
        results = model(frame, conf = 0.90)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

cap.release()
cv2.destroyAllWindows()