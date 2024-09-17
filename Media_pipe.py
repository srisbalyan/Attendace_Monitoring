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
import csv
from datetime import datetime
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Camera sources configurations
CAMERA_SOURCES = {
    "Webcam": 0,
    "Camera 1": "rtsp://admin:Nimda@2024@10.10.116.70:554/media/video1",
    "Camera 2": "rtsp://admin:Nimda@2024@10.10.116.71:554/media/video1",
    "Camera 3": "rtsp://admin:Nimda@2024@10.10.116.72:554/media/video1",
    "Camera 4": "rtsp://admin:Nimda@2024@10.10.116.73:554/media/video1",
    "Camera 5": "rtsp://admin:Nimda@2024@10.10.116.74:554/media/video1",
    "Camera 6": "rtsp://admin:Nimda@2024@10.10.116.75:554/media/video1"
}

# Constants
ANALYSIS_FREQUENCY = config.getint('Settings', 'ANALYSIS_FREQUENCY', fallback=30)
CONFIDENCE_THRESHOLD = config.getfloat('Settings', 'CONFIDENCE_THRESHOLD', fallback=0.5)
MIN_FACE_SIZE = config.getint('Settings', 'MIN_FACE_SIZE', fallback=10)  # Minimum size for a face to be considered

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
            try:
                for model in recognition_models:
                    embedding = DeepFace.represent(img_path, model_name=model, enforce_detection=False)[0]["embedding"]
                    known_faces[model][name] = embedding
                logging.info(f"Processed {img_path}")
            except Exception as e:
                logging.error(f"Error processing {img_path}: {str(e)}")

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

def determine_attentiveness(emotion):
    attentive_emotions = ['neutral', 'happy', 'surprise']
    return "Attentive" if emotion in attentive_emotions else "Not Attentive"

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

def resize_frame(frame, max_width=1280, max_height=720):
    height, width = frame.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return frame

class AttendanceTracker:
    def __init__(self):
        self.attendance_data = {}

    def update(self, name, timestamp, mood):
        if name != "Unknown":  # Only track recognized faces
            if name not in self.attendance_data:
                self.attendance_data[name] = {
                    'first_seen': timestamp, 
                    'last_seen': timestamp, 
                    'total_time': 0,
                    'moods': [mood]
                }
            else:
                last_seen = self.attendance_data[name]['last_seen']
                time_diff = (timestamp - last_seen).total_seconds()
                self.attendance_data[name]['total_time'] += time_diff
                self.attendance_data[name]['last_seen'] = timestamp
                self.attendance_data[name]['moods'].append(mood)


    def save_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'First Seen', 'Last Seen', 'Total Time (seconds)', 'Dominant Mood'])
            for name, data in self.attendance_data.items():
                dominant_mood = max(set(data['moods']), key=data['moods'].count) if data['moods'] else 'Unknown'
                writer.writerow([
                    name, 
                    data['first_seen'].strftime('%Y-%m-%d %H:%M:%S'),
                    data['last_seen'].strftime('%Y-%m-%d %H:%M:%S'),
                    round(data['total_time'], 2),
                    dominant_mood
                ])
        print(f"Attendance data saved to {filename}")

class FaceRecognitionApp:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.frame = None
        self.face_data = []
        self.prev_time = 0
        self.frame_count = 0
        self.last_detection_time = 0
        self.detection_interval = 0.5
        self.current_model = list(deepface_models.keys())[0]
        self.attendance_tracker = AttendanceTracker()
        self.face_recognition_history = {}
        self.confidence_threshold = 0.7
        self.min_face_size = MIN_FACE_SIZE
        self.current_camera = "Webcam"
        
        # Create the layout after setting all attributes
        layout = self.create_layout()
        self.window = sg.Window('Face Recognition System', layout, finalize=True)

    def create_layout(self):
        model_choices = ["All Models"] + list(deepface_models.keys())
        camera_choices = list(CAMERA_SOURCES.keys())
        return [
            [sg.Image(filename='', key='-IMAGE-')],
            [sg.Text('Face Count: 0', key='-COUNT-'), sg.Text('FPS: 0.00', key='-FPS-')],
            [sg.Text('Analysis Frequency:'), sg.Slider(range=(1, 60), default_value=ANALYSIS_FREQUENCY, orientation='h', key='-ANALYSIS_FREQ-')],
            [sg.Text('Confidence Threshold:'), sg.Slider(range=(0.1, 1.0), default_value=self.confidence_threshold, orientation='h', resolution=0.1, key='-CONF_THRESHOLD-')],
            [sg.Text('Minimum Face Size:'), sg.Slider(range=(10, 200), default_value=self.min_face_size, orientation='h', key='-MIN_FACE_SIZE-')],
            [sg.Text('Select Camera:'), sg.Combo(camera_choices, default_value=self.current_camera, key='-CAMERA-', enable_events=True)],
            [sg.Text('Select Recognition Model:'), sg.Combo(model_choices, default_value="All Models", key='-MODEL-', enable_events=True)],
            [sg.Button('Start'), sg.Button('Stop'), sg.Button('Save Attendance'), sg.Button('Reload Known Faces'), sg.Button('Exit')],
            [sg.Button('Toggle Mood Display'), sg.Button('Toggle Attentiveness Display')]
        ]

    def capture_frames(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = resize_frame(frame)  # Resize the frame
            else:
                logging.error("Failed to capture frame")
                break

    def reload_known_faces(self):
        global known_faces
        known_faces = load_known_faces("known_faces_images")
        sg.popup(f"Reloaded {len(known_faces)} known faces")

    def recognize_face_with_history(self, face_img, face_id):
        name, confidence = ensemble_face_recognition(face_img, self.current_model)
        
        if face_id not in self.face_recognition_history:
            self.face_recognition_history[face_id] = deque(maxlen=10)
        
        self.face_recognition_history[face_id].append((name, confidence))
        
        # Voting system
        names, confidences = zip(*self.face_recognition_history[face_id])
        voted_name = max(set(names), key=names.count)
        avg_confidence = sum(conf for n, conf in self.face_recognition_history[face_id] if n == voted_name) / names.count(voted_name)
        
        return voted_name, avg_confidence

    def run(self):
        show_mood = True
        show_attentiveness = True
        while True:
            event, values = self.window.read(timeout=20)
            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break

            if event == 'Start' and not self.is_running:
                self.current_camera = values['-CAMERA-']
                camera_source = CAMERA_SOURCES[self.current_camera]
                self.cap = cv2.VideoCapture(camera_source)
                if not self.cap.isOpened():
                    sg.popup(f"Error: Could not open {self.current_camera}")
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

            if event == '-CAMERA-':
                if self.is_running:
                    sg.popup("Please stop the current session before changing the camera.")
                else:
                    self.current_camera = values['-CAMERA-']

            if event == 'Save Attendance':
                filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.attendance_tracker.save_to_csv(filename)
                sg.popup(f"Attendance saved to {filename}")

            if event == 'Reload Known Faces':
                self.reload_known_faces()
            
            if event == 'Toggle Mood Display':
                show_mood = not show_mood

            if event == 'Toggle Attentiveness Display':
                show_attentiveness = not show_attentiveness

            if self.is_running and self.frame is not None:
                try:
                    self.frame_count += 1
                    current_time = time.time()
                    fps = 1 / (current_time - self.prev_time)
                    self.prev_time = current_time

                    self.confidence_threshold = float(values['-CONF_THRESHOLD-'])
                    self.min_face_size = int(values['-MIN_FACE_SIZE-'])
                    analysis_interval = int(values['-ANALYSIS_FREQ-'])

                    frame_with_boxes = self.frame.copy()

                    if current_time - self.last_detection_time >= self.detection_interval:
                        self.last_detection_time = current_time
                        detected_faces = detect_faces(self.frame)
                        self.face_data = []
                        logging.info(f"Detected {len(detected_faces)} faces")
                        for face_id, face in enumerate(detected_faces):
                            try:
                                x1, y1, x2, y2, conf = face
                                if conf > self.confidence_threshold and (x2-x1) >= self.min_face_size and (y2-y1) >= self.min_face_size:
                                    face_img = self.frame[y1:y2, x1:x2]
                                    if face_img.size > 0:
                                        name, confidence = self.recognize_face_with_history(face_img, face_id)
                                        emotion_analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                                        emotion = emotion_analysis[0]['dominant_emotion'] if emotion_analysis else 'Unknown'
                                        attentiveness = determine_attentiveness(emotion)
                                        self.face_data.append({
                                            'bbox': (x1, y1, x2-x1, y2-y1),
                                            'name': name,
                                            'confidence': confidence,
                                            'emotion': emotion,
                                            'attentiveness': attentiveness
                                        })
                                        # Update attendance tracker
                                        self.attendance_tracker.update(name, datetime.now(), emotion)
                            except Exception as e:
                                logging.error(f"Error processing detected face: {str(e)}")
                        logging.info(f"Processed {len(self.face_data)} faces after filtering")
                        print(f"Current attendance data: {self.attendance_tracker.attendance_data}")

                    for face in self.face_data:
                        x, y, w, h = face['bbox']
                        cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{face['name']} ({face['confidence']:.2f})"
                        cv2.putText(frame_with_boxes, label, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if show_mood:
                            cv2.putText(frame_with_boxes, f"Mood: {face['emotion']}", (x, y+h+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if show_attentiveness:
                            cv2.putText(frame_with_boxes, f"Attentiveness: {face['attentiveness']}", (x, y+h+40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
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