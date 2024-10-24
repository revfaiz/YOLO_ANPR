
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
