# #1
# import os
# import cv2
# import numpy as np
# from flask import Flask, Response, request, jsonify
# from ultralytics import YOLO
# import easyocr

# app = Flask(__name__)

# class CarDetector:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)

#     def detect(self, frame):
#         results = self.model(frame)
#         return results

# class LicensePlateDetector:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)

#     def detect(self, frame):
#         results = self.model(frame)
#         return results

# class VideoProcessor:
#     def __init__(self, car_model_path, license_model_path, video_source):
#         self.car_detector = CarDetector(car_model_path)
#         self.license_plate_detector = LicensePlateDetector(license_model_path)
#         self.ocr_reader = easyocr.Reader(['en'])
#         self.cap = cv2.VideoCapture(video_source)

#     def generate_frames(self):
#         while True:
#             success, frame = self.cap.read()
#             if not success:
#                 break

#             # Car detection
#             car_results = self.car_detector.detect(frame)
#             # License plate detection
#             license_results = self.license_plate_detector.detect(frame)
#             # Overlay detections
#             frame = self.overlay_detections(frame, car_results, license_results)

#             # Encode the frame as JPEG
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     def overlay_detections(self, frame, car_results, license_results):
#         for box in car_results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             label = 'Car'
#             frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#         for idx, box in enumerate(license_results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             plate_region = frame[y1:y2, x1:x2]
#             ocr_results = self.ocr_reader.readtext(plate_region)

#             plate_text = 'Unknown Plate'
#             if ocr_results:
#                 plate_text = ' '.join([text[1] for text in ocr_results])

#             label = f'License Plate: {plate_text}'
#             frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

#             white_background = 255 * np.ones(shape=[50, x2 - x1, 3], dtype=np.uint8)
#             cv2.putText(white_background, plate_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#             if y1 - 50 >= 0:
#                 frame[y1 - 50:y1, x1:x2] = white_background

#         return frame

# @app.route('/')
# def index():
#     return '''
#     <h1>Live Car and License Plate Detection</h1>
#     <input type="text" id="videoPath" placeholder="Enter video file path" style="width: 300px;">
#     <button onclick="startFeed()">Start Feed</button>
#     <img id="videoStream" src="" style="width: 100%; height: auto; display: none;">
#     <script>
#         function startFeed() {
#             const videoPath = document.getElementById('videoPath').value;
#             fetch('/video_feed', {
#                 method: 'POST',
#                 headers: {
#                     'Content-Type': 'application/json'
#                 },
#                 body: JSON.stringify({ video_path: videoPath })
#             }).then(response => {
#                 if (response.ok) {
#                     document.getElementById('videoStream').src = '/video_feed';
#                     document.getElementById('videoStream').style.display = 'block';
#                 } else {
#                     alert('Error: ' + response.statusText);
#                 }
#             });
#         }
#     </script>
#     '''

# @app.route('/video_feed', methods=['POST'])
# def video_feed():
#     data = request.get_json()
#     video_path = data.get('video_path')

#     if not os.path.exists(video_path):
#         return jsonify({'error': 'Video path does not exist'}), 400

#     car_model_path = 'D:/YOLO_ANPR/Anpr4/plate_test/yolov8n.pt'
#     license_model_path = 'D:/YOLO_ANPR/Anpr4/plate_test/license_plate_detector.pt'

#     video_processor = VideoProcessor(car_model_path, license_model_path, video_path)
#     return Response(video_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)

#2
import os
import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template
from ultralytics import YOLO
import easyocr

app = Flask(__name__)

# Path to save uploaded and processed videos
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    def __init__(self, car_model_path, license_model_path, video_source=0):
        self.car_detector = CarDetector(car_model_path)
        self.license_plate_detector = LicensePlateDetector(license_model_path)
        self.video_source = video_source
        self.ocr_reader = easyocr.Reader(['en'])
        self.cap = cv2.VideoCapture(self.video_source)

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
    return Response(video_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed',methods=['GET'])
def video_feed():
    return Response(video_processor.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    car_model_path = 'D:/YOLO_ANPR/Anpr4/plate_test/yolov8n.pt'
    license_model_path = 'D:\YOLO ANPR\Anpr4\plate_test\license_plate_detector.pt'
    video_processor = VideoProcessor(car_model_path, license_model_path)
    app.run(debug=True, host='0.0.0.0', port=5000)

#3
# import os
# import cv2
# import torch
# import easyocr
# import numpy as np
# from flask import Flask, request, jsonify, send_file
# from ultralytics import YOLO

# app = Flask(__name__)

# # Paths to save uploaded and processed videos
# UPLOAD_FOLDER = './uploads'
# PROCESSED_FOLDER = './processed'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# class CarDetector:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)

#     def detect(self, frame):
#         results = self.model(frame)
#         return results

# class LicensePlateDetector:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)

#     def detect(self, frame):
#         results = self.model(frame)
#         return results

# class VideoProcessor:
#     def __init__(self, car_model_path, license_model_path, input_video_path, output_video_path):
#         self.car_detector = CarDetector(car_model_path)
#         self.license_plate_detector = LicensePlateDetector(license_model_path)
#         self.input_video_path = input_video_path
#         self.output_video_path = output_video_path
#         self.ocr_reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

#     def process_video(self):
#         cap = cv2.VideoCapture(self.input_video_path)
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(self.output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Car detection
#             car_results = self.car_detector.detect(frame)
#             # License plate detection
#             license_results = self.license_plate_detector.detect(frame)
            
#             # Overlay detections (bounding boxes, labels)
#             frame = self.overlay_detections(frame, car_results, license_results)

#             out.write(frame)

#         cap.release()
#         out.release()

#     def overlay_detections(self, frame, car_results, license_results):
#         for box in car_results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             label = 'Car'
#             frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#         for idx, box in enumerate(license_results[0].boxes):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             # Extract the region for OCR
#             plate_region = frame[y1:y2, x1:x2]
#             ocr_results = self.ocr_reader.readtext(plate_region)

#             # Default text if OCR fails
#             plate_text = 'Unknown Plate'

#             if ocr_results:
#                 plate_text = ' '.join([text[1] for text in ocr_results])

#             label = f'License Plate: {plate_text}'
#             frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)  # Change to black

#             # Create a white background for the cropped plate image
#             white_background = 255 * np.ones(shape=[50, x2 - x1, 3], dtype=np.uint8)

#             # Put the text on the white background
#             cv2.putText(white_background, plate_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#             # Ensure the placement coordinates are valid
#             if y1 - 50 >= 0:
#                 frame[y1 - 50:y1, x1:x2] = white_background  # Place it above the detection box

#         return frame


# # Define Flask route to process video
# @app.route('/process_video', methods=['POST'])
# def process_video_route():
#     data = request.get_json()

#     if 'video_path' not in data:
#         return jsonify({'error': 'No video path provided'}), 400

#     video_path = data['video_path']

#     if not os.path.exists(video_path):
#         return jsonify({'error': 'Video path does not exist'}), 400

#     # Define paths for models and output video
#     car_model_path = 'D:/YOLO_ANPR/Anpr4/plate_test/yolov8n.pt'
#     license_model_path = 'D:\YOLO ANPR\Anpr4\plate_test\license_plate_detector.pt'
#     output_video_path = os.path.join(PROCESSED_FOLDER, f'processed_{os.path.basename(video_path)}')

#     # Process the video
#     video_processor = VideoProcessor(car_model_path, license_model_path, video_path, output_video_path)
#     video_processor.process_video()

#     # Send the processed video back as a response
#     return send_file(output_video_path, as_attachment=True)

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)
