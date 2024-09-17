from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU acceleration is available")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("GPU acceleration is not available, using CPU")

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
MIN_FACE_SIZE = config.getint('Settings', 'MIN_FACE_SIZE', fallback=10)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD, model_selection=1)

# Initialize DeepFace recognition models
recognition_models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'ArcFace', 'Dlib']
deepface_models = {}
for model in recognition_models:
    try:
        deepface_models[model] = DeepFace.build_model(model)
        logging.info(f"Successfully loaded {model} model")
    except Exception as e:
        logging.error(f"Failed to load {model} model: {str(e)}")

if not deepface_models:
    raise ValueError("No face recognition models could be loaded. Please check your DeepFace installation.")

# Global variables
is_running = False
current_camera = "Webcam"
current_model = list(deepface_models.keys())[0] if deepface_models else None
show_mood = True
show_attentiveness = True
cap = None
frame = None
face_data = []
prev_time = 0
frame_count = 0
last_detection_time = 0
detection_interval = 0.5
confidence_threshold = 0.7
min_face_size = MIN_FACE_SIZE
attendance_tracker = None
face_recognition_history = {}

def load_known_faces(directory):
    """
    Load known faces from a directory and cache them for faster loading in future.
    """
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
    """
    Detect faces in a frame using MediaPipe Face Detection.
    """
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
    """
    Recognize a face using a specific DeepFace model.
    """
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
    """
    Determine if a person is attentive based on their emotion.
    """
    attentive_emotions = ['neutral', 'happy', 'surprise']
    return "Attentive" if emotion in attentive_emotions else "Not Attentive"

def ensemble_face_recognition(face_img, selected_model=None):
    """
    Perform ensemble face recognition using multiple models or a selected model.
    """
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
    """
    Resize a frame to fit within specified dimensions while maintaining aspect ratio.
    """
    height, width = frame.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return frame

class AttendanceTracker:
    """
    Track attendance and mood of recognized individuals.
    """
    def __init__(self):
        self.attendance_data = {}

    def update(self, name, timestamp, mood):
        if name != "Unknown":
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

class CustomEnsemble:
    """
    Custom ensemble model for face recognition.
    """
    def __init__(self, models, known_faces_dir):
        self.models = models
        self.known_faces_dir = known_faces_dir
        self.embeddings = {model: {} for model in models}
        self.weights = {model: 1 for model in models}
        self.label_to_name = {}
        self.name_to_label = {}

    def prepare_dataset(self, progress_callback=None):
        X = {model: [] for model in self.models}
        y = []
        label = 0
        total_files = len([f for f in os.listdir(self.known_faces_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        for i, filename in enumerate(os.listdir(self.known_faces_dir)):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                parts = filename.split('_')
                if len(parts) == 3:
                    name = '_'.join(parts[:-2])
                elif len(parts) == 2:
                    name = parts[0]
                else:
                    continue
                
                img_path = os.path.join(self.known_faces_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if name not in self.name_to_label:
                    self.name_to_label[name] = label
                    self.label_to_name[label] = name
                    label += 1
                
                y.append(self.name_to_label[name])
                
                for model_name in self.models:
                    try:
                        embedding = DeepFace.represent(img, model_name=model_name, enforce_detection=False)[0]["embedding"]
                        X[model_name].append(embedding)
                        if name not in self.embeddings[model_name]:
                            self.embeddings[model_name][name] = []
                        self.embeddings[model_name][name].append(embedding)
                    except Exception as e:
                        logging.error(f"Error generating embedding for {filename} with model {model_name}: {str(e)}")
                
                if progress_callback:
                    progress_callback(1)
            
            if i % 10 == 0:
                if progress_callback:
                    progress_callback(0)

        for model_name in self.models:
            for name in self.embeddings[model_name]:
                self.embeddings[model_name][name] = np.mean(self.embeddings[model_name][name], axis=0)
        
        return X, np.array(y)

    def train(self, progress_callback=None):
        X, y = self.prepare_dataset(progress_callback)
        
        for model_name in self.models:
            X_train, X_test, y_train, y_test = train_test_split(X[model_name], y, test_size=0.2, random_state=42)
            
            y_pred = []
            for embedding in X_test:
                predicted_name = self.find_closest_match(embedding, model_name)
                y_pred.append(self.name_to_label[predicted_name])
                
                if progress_callback:
                    progress_callback(1)
            
            accuracy = accuracy_score(y_test, y_pred)
            self.weights[model_name] = accuracy
        
        total_weight = sum(self.weights.values())
        for model_name in self.weights:
            self.weights[model_name] /= total_weight

    def find_closest_match(self, embedding, model_name):
        min_distance = float('inf')
        closest_name = "Unknown"
        for name, known_embedding in self.embeddings[model_name].items():
            distance = np.linalg.norm(embedding - known_embedding)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        return closest_name

    def predict(self, face_img):
        votes = {}
        for model_name in self.models:
            try:
                embedding = DeepFace.represent(face_img, model_name=model_name, enforce_detection=False)[0]["embedding"]
                name = self.find_closest_match(embedding, model_name)
                if name not in votes:
                    votes[name] = 0
                votes[name] += self.weights[model_name]
            except Exception as e:
                logging.error(f"Error predicting with model {model_name}: {str(e)}")
        
        if not votes:
            return "Unknown", 0.0
        
        predicted_name = max(votes, key=votes.get)
        confidence = votes[predicted_name] / sum(votes.values())
        return predicted_name, confidence

def recognize_face_with_history(face_img, face_id):
    global face_recognition_history, current_model
    
    name, confidence = ensemble_face_recognition(face_img, current_model)
    
    if face_id not in face_recognition_history:
        face_recognition_history[face_id] = deque(maxlen=10)
    
    face_recognition_history[face_id].append((name, confidence))
    
    # Voting system
    names, confidences = zip(*face_recognition_history[face_id])
    voted_name = max(set(names), key=names.count)
    avg_confidence = sum(conf for n, conf in face_recognition_history[face_id] if n == voted_name) / names.count(voted_name)
    
    return voted_name, avg_confidence

def gen_frames():
    global frame, is_running, cap, face_data, prev_time, frame_count, last_detection_time

    while is_running:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = resize_frame(frame)
            current_time = time.time()
            
            if current_time - last_detection_time >= detection_interval:
                last_detection_time = current_time
                detected_faces = detect_faces(frame)
                face_data = []
                for face_id, face in enumerate(detected_faces):
                    try:
                        x1, y1, x2, y2, conf = face
                        if conf > confidence_threshold and (x2-x1) >= min_face_size and (y2-y1) >= min_face_size:
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size > 0:
                                name, confidence = recognize_face_with_history(face_img, face_id)
                                emotion_analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                                emotion = emotion_analysis[0]['dominant_emotion'] if emotion_analysis else 'Unknown'
                                attentiveness = determine_attentiveness(emotion)
                                face_data.append({
                                    'bbox': (x1, y1, x2-x1, y2-y1),
                                    'name': name,
                                    'confidence': confidence,
                                    'emotion': emotion,
                                    'attentiveness': attentiveness
                                })
                                # Update attendance tracker
                                attendance_tracker.update(name, datetime.now(), emotion)
                    except Exception as e:
                        logging.error(f"Error processing detected face: {str(e)}")

            # Draw bounding boxes and labels
            for face in face_data:
                x, y, w, h = face['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{face['name']} ({face['confidence']:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if show_mood:
                    cv2.putText(frame, f"Mood: {face['emotion']}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if show_attentiveness:
                    cv2.putText(frame, f"Attentiveness: {face['attentiveness']}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def train_custom_ensemble():
    global custom_ensemble
    try:
        custom_ensemble = CustomEnsemble(recognition_models, "known_faces_images")
        custom_ensemble.prepare_dataset()
        custom_ensemble.train()
        return jsonify({"status": "success", "message": "Custom ensemble training completed!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during custom ensemble training: {str(e)}"})

@app.route('/')
def index():
    return render_template('index.html', camera_sources=CAMERA_SOURCES, recognition_models=recognition_models)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global is_running, cap, current_camera
    if not is_running:
        try:
            current_camera = request.form['camera']
            camera_source = CAMERA_SOURCES[current_camera]
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                logging.error(f"Could not open {current_camera}")
                return jsonify({"status": "error", "message": f"Could not open {current_camera}"})
            is_running = True
            logging.info(f"Face recognition started with camera: {current_camera}")
            return jsonify({"status": "success", "message": "Face recognition started"})
        except Exception as e:
            logging.error(f"Error starting face recognition: {str(e)}")
            return jsonify({"status": "error", "message": f"Error starting face recognition: {str(e)}"})
    return jsonify({"status": "error", "message": "Face recognition is already running"})

@app.route('/stop', methods=['POST'])
def stop():
    global is_running, cap
    if is_running:
        is_running = False
        if cap:
            cap.release()
        logging.info("Face recognition stopped")
        return jsonify({"status": "success", "message": "Face recognition stopped"})
    logging.info("Attempted to stop face recognition, but it was not running")
    return jsonify({"status": "error", "message": "Face recognition is not running"})

@app.route('/save_attendance', methods=['POST'])
def save_attendance():
    global attendance_tracker
    if attendance_tracker:
        filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        attendance_tracker.save_to_csv(filename)
        return jsonify({"status": "success", "message": f"Attendance saved to {filename}"})
    return jsonify({"status": "error", "message": "No attendance data available"})

@app.route('/reload_known_faces', methods=['POST'])
def reload_known_faces():
    global known_faces
    known_faces = load_known_faces("known_faces_images")
    return jsonify({"status": "success", "message": f"Reloaded {len(known_faces)} known faces"})

@app.route('/toggle_mood', methods=['POST'])
def toggle_mood():
    global show_mood
    show_mood = not show_mood
    return jsonify({"status": "success", "show_mood": show_mood})

@app.route('/toggle_attentiveness', methods=['POST'])
def toggle_attentiveness():
    global show_attentiveness
    show_attentiveness = not show_attentiveness
    return jsonify({"status": "success", "show_attentiveness": show_attentiveness})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global confidence_threshold, min_face_size, current_model
    confidence_threshold = float(request.form['confidence_threshold'])
    min_face_size = int(request.form['min_face_size'])
    current_model = request.form['recognition_model']
    return jsonify({"status": "success", "message": "Settings updated"})

@app.route('/train_custom_ensemble', methods=['POST'])
def train_custom_ensemble_route():
    return train_custom_ensemble()

if __name__ == '__main__':
    app.run(debug=True)
