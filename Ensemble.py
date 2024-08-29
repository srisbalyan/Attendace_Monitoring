import cv2
import numpy as np
from deepface import DeepFace
import PySimpleGUI as sg
import time
import os
import threading
import csv
from datetime import datetime
import configparser
import pickle
import logging
import concurrent.futures
from ultralytics import YOLO
import tensorflow as tf
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
ANALYSIS_FREQUENCY = 30
CONFIDENCE_THRESHOLD = 0.6

logging.info("Initializing YOLO models...")
yolo_fast = YOLO("yolov8n-face.pt")
yolo_accurate = YOLO("yolov8s-face.pt")
logging.info("YOLO models initialized")

# Initialize DeepFace models
logging.info("Initializing DeepFace models...")
deepface_models = ['VGG-Face', 'Facenet']  # Reduced number of models
models = {}
for model_name in deepface_models:
    try:
        logging.info(f"Loading {model_name}...")
        models[model_name] = DeepFace.build_model(model_name)
        logging.info(f"{model_name} loaded successfully")
    except Exception as e:
        logging.error(f"Error loading {model_name}: {str(e)}")
logging.info("DeepFace models initialization complete")

# Load known faces
def load_known_faces(directory):
    logging.info(f"Loading known faces from {directory}")
    known_faces = {model: {} for model in deepface_models}
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(directory, filename)
            logging.info(f"Processing {img_path}")
            for model in deepface_models:
                try:
                    embedding = DeepFace.represent(img_path, model_name=model, enforce_detection=False)[0]["embedding"]
                    known_faces[model][name] = np.array(embedding)  # Convert to numpy array
                    logging.info(f"Embedding created for {name} using {model}")
                except Exception as e:
                    logging.error(f"Error processing {img_path} with {model}: {str(e)}")
    logging.info("Known faces loading complete")
    return known_faces

known_faces = load_known_faces("known_faces_images")

# Face recognition function
def recognize_face(face_img, model_name):
    try:
        embedding = np.array(DeepFace.represent(face_img, model_name=model_name, enforce_detection=False)[0]["embedding"])
        min_distance = float('inf')
        recognized_name = "Unknown"
        for name, known_embedding in known_faces[model_name].items():
            distance = np.linalg.norm(embedding - known_embedding)
            if distance < min_distance:
                min_distance = distance
                recognized_name = name
        confidence = 1 / (1 + min_distance)
        return recognized_name, confidence
    except Exception as e:
        logging.error(f"Error in recognize_face: {str(e)}")
        return "Unknown", 0.0

# Ensemble recognition
def ensemble_recognition(face_img):
    results = []
    for model in deepface_models:
        name, confidence = recognize_face(face_img, model)
        results.append((name, confidence))
    
    names, confidences = zip(*results)
    voted_name = max(set(names), key=names.count)
    avg_confidence = sum(conf for name, conf in results if name == voted_name) / names.count(voted_name)
    
    return voted_name, avg_confidence

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
        self.detection_interval = 1.0
        self.trackers = []

    def create_layout(self):
        return [
            [sg.Image(filename='', key='-IMAGE-')],
            [sg.Text('Face Count: 0', key='-COUNT-'), sg.Text('FPS: 0.00', key='-FPS-')],
            [sg.Text('Detection Model:'), sg.Radio('YOLO Fast', 'RADIO1', default=True, key='-YOLO-FAST-'),
             sg.Radio('YOLO Accurate', 'RADIO1', key='-YOLO-ACCURATE-')],
            [sg.Text('Analysis Frequency:'), sg.Slider(range=(1, 60), default_value=ANALYSIS_FREQUENCY, orientation='h', key='-ANALYSIS_FREQ-')],
            [sg.Text('Confidence Threshold:'), sg.Slider(range=(0.1, 1.0), default_value=CONFIDENCE_THRESHOLD, orientation='h', resolution=0.1, key='-CONF_THRESHOLD-')],
            [sg.Button('Start'), sg.Button('Stop'), sg.Button('Exit')]
        ]

    def capture_frames(self):
        while self.is_running:
            ret, self.frame = self.cap.read()
            if not ret:
                break

    def detect_faces(self, frame):
        if self.window['-YOLO-FAST-'].get():
            results = yolo_fast(frame)
        else:
            results = yolo_accurate(frame)
        return [(box.xyxy[0].tolist(), box.conf.item()) for box in results[0].boxes]

    def process_face(self, face_img):
        name, confidence = ensemble_recognition(face_img)
        return name, confidence

    def run(self):
        while True:
            event, values = self.window.read(timeout=20)
            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break

            if event == 'Start' and not self.is_running:
                self.cap = cv2.VideoCapture(0)
                self.is_running = True
                threading.Thread(target=self.capture_frames, daemon=True).start()

            if event == 'Stop':
                self.is_running = False
                if self.cap:
                    self.cap.release()

            if self.is_running and self.frame is not None:
                current_time = time.time()
                fps = 1 / (current_time - self.prev_time)
                self.prev_time = current_time

                frame_with_boxes = self.frame.copy()

                self.trackers = [tracker for tracker in self.trackers if tracker.update(self.frame)[0]]

                if current_time - self.last_detection_time >= self.detection_interval:
                    self.last_detection_time = current_time
                    detected_faces = self.detect_faces(self.frame)
                    self.trackers = []
                    self.face_data = []

                    for (x1, y1, x2, y2), conf in detected_faces:
                        if conf > values['-CONF_THRESHOLD-']:
                            face_img = self.frame[int(y1):int(y2), int(x1):int(x2)]
                            name, recognition_conf = self.process_face(face_img)
                            self.face_data.append({
                                'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                                'name': name,
                                'confidence': recognition_conf
                            })
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(self.frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
                            self.trackers.append(tracker)

                for face, tracker in zip(self.face_data, self.trackers):
                    success, bbox = tracker.update(self.frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame_with_boxes, f"{face['name']} ({face['confidence']:.2f})",
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                imgbytes = cv2.imencode('.png', frame_with_boxes)[1].tobytes()
                self.window['-IMAGE-'].update(data=imgbytes)
                self.window['-COUNT-'].update(f'Face Count: {len(self.face_data)}')
                self.window['-FPS-'].update(f'FPS: {fps:.2f}')

        self.window.close()
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    logging.info("Starting Face Recognition App")
    app = FaceRecognitionApp()
    app.run()