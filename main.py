# from ultralytics import YOLO
# import cv2
# import numpy as np
# from sort.sort import Sort
# from util import get_car, read_license_plate
# from visualize import visualize

# # Initialize models and trackers
# coco_model = YOLO('yolov8n.pt')  # Load the vehicle detection model
# license_plate_detector = YOLO('./license_plate_detector.pt')  # Load the license plate detection model
# mot_tracker = Sort()  # Tracker for tracking detected cars

# # Load video source (replace with camera feed if required: cap = cv2.VideoCapture(0))
# # input = "rtsp://admin:DHA@1431@192.168.80.250:554/Streaming/Channels/101"
# cap = cv2.VideoCapture('../car.mp4')

# vehicles = [2, 3, 5, 7]  # Class IDs for vehicles (e.g., car, bus, truck)

# # Read frames in real-time
# frame_nmr = -1
# ret = True

# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Dictionary to store all tracked vehicles and their license plate data in this frame
#     frame_car_data = {}

#     # Detect vehicles in the current frame
#     detections = coco_model(frame)[0]
#     detections_ = []
#     for detection in detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) in vehicles:
#             detections_.append([x1, y1, x2, y2, score])

#     # Track vehicles using SORT
#     track_ids = mot_tracker.update(np.asarray(detections_))

#     # Detect license plates in the current frame
#     license_plates = license_plate_detector(frame)[0]
#     for license_plate in license_plates.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = license_plate

#         # Associate detected license plate with a tracked vehicle
#         xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

#         if car_id != -1:
#             # Crop and process the license plate image
#             license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#             license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#             _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

#             # Read license plate text using OCR
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

#             # Store results for this car in the current frame
#             if license_plate_text is not None:
#                 frame_car_data[car_id] = {
#                     'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                     'license_plate': {
#                         'bbox': [x1, y1, x2, y2],
#                         'text': license_plate_text,
#                         'bbox_score': score,
#                         'text_score': license_plate_text_score
#                     }
#                 }


#     print("-------------------------------------",frame_car_data)
#     # Pass the entire dictionary of detected cars and license plates to `visualize`
#     visualize(frame_car_data, video_source=frame)

# cap.release()
# cv2.destroyAllWindows()

#2
import os
import cv2
import numpy as np
from flask import Flask, Response,request,jsonify
from ultralytics import YOLO
import easyocr

app = Flask(__name__)

# Paths for the models
car_model_path = 'D:/YOLO_ANPR/Anpr4/plate_test/yolov8n.pt'
license_model_path = 'D:\YOLO ANPR\Anpr4\plate_test\license_plate_detector.pt'

class CarDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        return results

class LicensePlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        return results

class VideoProcessor:
    def __init__(self, car_model_path, license_model_path, video_path):
        self.car_detector = CarDetector(car_model_path)
        self.license_plate_detector = LicensePlateDetector(license_model_path)
        self.video_path = video_path
        self.ocr_reader = easyocr.Reader(['en'])
        self.cap = cv2.VideoCapture(self.video_path)

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # Car detection
            car_results = self.car_detector.detect(frame)
            # License plate detection
            license_results = self.license_plate_detector.detect(frame)
            # Overlay detections
            frame = self.overlay_detections(frame, car_results, license_results)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def overlay_detections(self, frame, car_results, license_results):
        for box in car_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = 'Car'
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        for idx, box in enumerate(license_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_region = frame[y1:y2, x1:x2]
            ocr_results = self.ocr_reader.readtext(plate_region)

            plate_text = 'Unknown Plate'
            if ocr_results:
                plate_text = ' '.join([text[1] for text in ocr_results])

            label = f'License Plate: {plate_text}'
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            white_background = 255 * np.ones(shape=[50, x2 - x1, 3], dtype=np.uint8)
            cv2.putText(white_background, plate_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if y1 - 50 >= 0:
                frame[y1 - 50:y1, x1:x2] = white_background

        return frame

@app.route('/')
def index():
    return "<h1>Video Feed</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    video_path = r"UPLOAD_FOLDER\demo.mp4"
    video_processor = VideoProcessor(car_model_path, license_model_path, video_path)
    return Response(video_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0', port=5000)
