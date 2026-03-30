from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")
image_path='./image.jpeg'

def detect_person(image_path):
    results = yolo(image_path)
    return results[0].boxes.xyxy  # bounding box


print(detect_person(image_path))