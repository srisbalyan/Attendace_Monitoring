import cv2
import numpy as np
from deepface import DeepFace
import PySimpleGUI as sg
import time
import os
import threading
import configparser
import pickle
import logging
import mediapipe as mp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Constants
ANALYSIS_FREQUENCY = config.getint('Settings', 'ANALYSIS_FREQUENCY', fallback=30)
CONFIDENCE_THRESHOLD = config.getfloat('Settings', 'CONFIDENCE_THRESHOLD', fallback=0.5)
MIN_FACE_SIZE = 10 # Minimum size for a face to be considered

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD,model_selection=1)

# Initialize DeepFace recognition models
recognition_models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'ArcFace', 'Dlib']
deepface_models = {}
for model in recognition_models:
    try:
        deepface_models[model] = DeepFace.build_model(model)
        logging.info(f"Successfully loaded {model} model")
    except Exception as e:
        logging.error(f"Failed to load {model} model: {str(e)}")

def load_known_faces(directory):
    cache_file = 'known_faces_cache.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    known_faces = {model: {} for model in recognition_models}
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(directory, filename)
            for model in recognition_models:
                embedding = DeepFace.represent(img_path, model_name=model, enforce_detection=False)[0]["embedding"]
                known_faces[model][name] = embedding

    with open(cache_file, 'wb') as f:
        pickle.dump(known_faces, f)

    return known_faces

known_faces = load_known_faces("known_faces_images")

def detect_faces(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    detected_faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            if w > MIN_FACE_SIZE and h > MIN_FACE_SIZE:
                detected_faces.append([x, y, x+w, y+h, detection.score[0]])
    return detected_faces

def recognize_face(face_img, model_name):
    embedding = DeepFace.represent(face_img, model_name=model_name, enforce_detection=False)[0]["embedding"]
    min_distance = float('inf')
    recognized_name = "Unknown"
    for name, known_embedding in known_faces[model_name].items():
        distance = np.linalg.norm(embedding - known_embedding)
        if distance < min_distance:
            min_distance = distance
            recognized_name = name
    confidence = 1 / (1 + min_distance)
    return recognized_name, confidence

def ensemble_face_recognition(face_img, selected_model=None):
    results = []
    if selected_model and selected_model in recognition_models:
        name, confidence = recognize_face(face_img, selected_model)
        results.append((name, confidence))
    else:
        for model in recognition_models:
            name, confidence = recognize_face(face_img, model)
            results.append((name, confidence))
    
    names, confidences = zip(*results)
    final_name = max(set(names), key=names.count)
    final_confidence = sum([conf for name, conf in results if name == final_name]) / names.count(final_name)
    
    return final_name, final_confidence

class FaceRecognitionApp:
    def __init__(self):
        self.window = sg.Window('Face Recognition System', self.create_layout(), finalize=True)
        self.cap = None
        self.is_running = False
        self.frame = None
        self.face_data = []
        self.prev_time = 0
        self.frame_count = 0
        self.last_detection_time = 0
        self.detection_interval = 0.5
        self.current_model = list(deepface_models.keys())[0]

    def create_layout(self):
        model_choices = ["All Models"] + list(deepface_models.keys())
        return [
            [sg.Image(filename='', key='-IMAGE-')],
            [sg.Text('Face Count: 0', key='-COUNT-'), sg.Text('FPS: 0.00', key='-FPS-')],
            [sg.Text('Analysis Frequency:'), sg.Slider(range=(1, 60), default_value=ANALYSIS_FREQUENCY, orientation='h', key='-ANALYSIS_FREQ-')],
            [sg.Text('Confidence Threshold:'), sg.Slider(range=(0.1, 1.0), default_value=CONFIDENCE_THRESHOLD, orientation='h', resolution=0.1, key='-CONF_THRESHOLD-')],
            [sg.Text('Select Recognition Model:'), sg.Combo(model_choices, default_value="All Models", key='-MODEL-', enable_events=True)],
            [sg.Button('Start'), sg.Button('Stop'), sg.Button('Exit')],
        ]

    def capture_frames(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                logging.error("Failed to capture frame")
                break

    def run(self):
        while True:
            event, values = self.window.read(timeout=20)
            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break

            if event == 'Start' and not self.is_running:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    sg.popup("Error: Could not open webcam")
                else:
                    self.is_running = True
                    threading.Thread(target=self.capture_frames, daemon=True).start()

            if event == 'Stop':
                self.is_running = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
            
            if event == '-MODEL-':
                self.current_model = None if values['-MODEL-'] == "All Models" else values['-MODEL-']

            if self.is_running and self.frame is not None:
                try:
                    self.frame_count += 1
                    current_time = time.time()
                    fps = 1 / (current_time - self.prev_time)
                    self.prev_time = current_time

                    conf_threshold = float(values['-CONF_THRESHOLD-'])
                    analysis_interval = int(values['-ANALYSIS_FREQ-'])

                    frame_with_boxes = self.frame.copy()

                    if current_time - self.last_detection_time >= self.detection_interval:
                        self.last_detection_time = current_time
                        detected_faces = detect_faces(self.frame)
                        self.face_data = []
                        logging.info(f"Detected {len(detected_faces)} faces")
                        for face in detected_faces:
                            try:
                                x1, y1, x2, y2, conf = face
                                if conf > conf_threshold:
                                    face_img = self.frame[y1:y2, x1:x2]
                                    if face_img.size > 0:
                                        name, confidence = ensemble_face_recognition(face_img, self.current_model)
                                        self.face_data.append({
                                            'bbox': (x1, y1, x2-x1, y2-y1),
                                            'name': name,
                                            'confidence': confidence
                                        })
                            except Exception as e:
                                logging.error(f"Error processing detected face: {str(e)}")
                        logging.info(f"Processed {len(self.face_data)} faces after filtering")

                    for face in self.face_data:
                        x, y, w, h = face['bbox']
                        cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{face['name']} ({face['confidence']:.2f})"
                        cv2.putText(frame_with_boxes, label, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # No need to convert BGR to RGB for PySimpleGUI
                    #frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
                    imgbytes = cv2.imencode('.png', frame_with_boxes)[1].tobytes()
                    self.window['-IMAGE-'].update(data=imgbytes)
                    self.window['-COUNT-'].update(f'Face Count: {len(self.face_data)}')
                    self.window['-FPS-'].update(f'FPS: {fps:.2f}')

                except Exception as e:
                    logging.error(f"Error processing frame: {str(e)}")

        self.window.close()
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()