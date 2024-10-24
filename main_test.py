from flaskapp import Flask, Response, request
import cv2
from flask_cors import CORS
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate
from visualize import visualize

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Initialize models and trackers
coco_model = YOLO('yolov8n.pt')  # Load the vehicle detection model
license_plate_detector = YOLO('./license_plate_detector.pt')  # Load the license plate detection model
mot_tracker = Sort()  # Tracker for tracking detected cars

# Set up the vehicle class IDs
vehicles = [2, 3, 5, 7]  # Class IDs for vehicles (e.g., car, bus, truck)


# cap = cv2.VideoCapture("rtsp://admin:DHA@1431@192.168.80.250:554/Streaming/Channels/101")
    

    # Define a generator to stream processed video frames
def generate():
        cap = cv2.VideoCapture("./car.mp4")
        frame_nmr = -1
        ret = True

        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if not ret:
                break

            # Dictionary to store all tracked vehicles and their license plate data in this frame
            frame_car_data = {}

            # Detect vehicles in the current frame
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles using SORT
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates in the current frame
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Associate detected license plate with a tracked vehicle
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Crop and process the license plate image
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate text using OCR
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    # Store results for this car in the current frame
                    if license_plate_text is not None:
                        frame_car_data[car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }

            # Visualize frame with bounding boxes and annotations (if needed)
            frame= visualize(frame_car_data, video_source=frame)

            # Encode the frame to JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame in byte format as a multipart response
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=False)
