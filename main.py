import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from utlies import get_car, read_license_plate, write_csv

# Initialize YOLOv8 and DeepSORT
coco_model = YOLO('yolov8n.pt')  
license_plate_detector = YOLO(r"C:\Users\Omar2\Downloads\project_1\LP-detection.pt")

deepsort = DeepSort()

# Load sample video
cap = cv2.VideoCapture(r"C:\Users\Omar2\Downloads\vehicles.mp4")

# Define vehicle class IDs (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
vehicles = {2, 3, 5, 7}

ret = True
results = {}
frame_nmr = 0

while ret:
    ret, frame = cap.read()
    if not ret or frame_nmr > 40: 
        break  

    results[frame_nmr] = {}
    
    # Detect vehicles
    detections = coco_model(frame)[0]
    detection_boxes = [([x1, y1, x2, y2], score, class_id) for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist() if int(class_id) in vehicles]
    
    # Track vehicles using DeepSORT
    tracks = deepsort.update_tracks(detection_boxes, frame=frame)
    
    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    
    for x1, y1, x2, y2, score, class_id in license_plates.boxes.data.tolist():
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # Get the corresponding vehicle
        xcar1, ycar1, xcar2, ycar2, car_id = get_car((x1, y1, x2, y2, score, class_id), tracks)
        if car_id != -1:
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            
            if license_plate_text:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }
    
    frame_nmr += 1

# Write results to CSV
write_csv(results, './test.csv')