import cv2
import numpy as np
from retinaface import RetinaFace
import dlib
import PySimpleGUI as sg
import pandas as pd
from datetime import datetime
import pickle
import time

# Load known faces
try:
    with open('known_faces.pkl', 'rb') as f:
        known_faces = pickle.load(f)
    print(f"Loaded {len(known_faces)} known faces.")
except FileNotFoundError:
    known_faces = {}
    print("No known faces found. The system will run in detection-only mode.")

# Initialize face detector
face_detector = RetinaFace.build_model()

# Initialize face recognizer
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to recognize face
def recognize_face(face_encoding):
    for name, known_encoding in known_faces.items():
        distance = np.linalg.norm(face_encoding - known_encoding)
        if distance < 0.6:  # Adjust threshold as needed
            return name
    return "Unknown"

# GUI Layout
layout = [
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Text('Face Count: 0', key='-COUNT-'), sg.Text('FPS: 0.00', key='-FPS-')],
    [sg.Multiline(size=(60, 10), key='-ATTENDANCE-', autoscroll=True)],
    [sg.Button('Start'), sg.Button('Stop'), sg.Button('Save Attendance'), sg.Button('Exit')]
]

window = sg.Window('Classroom Attendance System', layout, finalize=True)

# Initialize variables
cap = None
attendance_data = []
prev_time = 0

while True:
    event, values = window.read(timeout=20)
    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break

    if event == 'Start' and cap is None:
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    if event == 'Stop' and cap is not None:
        cap.release()
        cap = None

    if event == 'Save Attendance':
        if attendance_data:
            df = pd.DataFrame(attendance_data)
            filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            sg.popup(f"Attendance saved to {filename}")
        else:
            sg.popup("No attendance data to save.")

    if cap is not None:
        ret, frame = cap.read()
        if ret:
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Detect faces
            faces = RetinaFace.detect_faces(frame)
            
            # Process each detected face
            for face_key in faces:
                face = faces[face_key]
                facial_area = face['facial_area']
                x1, y1, x2, y2 = [int(coord) for coord in facial_area]
                
                # Extract face encoding
                face_img = frame[y1:y2, x1:x2]
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                shape = shape_predictor(rgb_face, dlib.rectangle(0, 0, x2-x1, y2-y1))
                face_encoding = np.array(face_recognizer.compute_face_descriptor(rgb_face, shape))
                
                # Recognize face
                name = recognize_face(face_encoding)
                
                # Draw rectangle and name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Update attendance
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                attendance_data.append({"Name": name, "Timestamp": timestamp})
                window['-ATTENDANCE-'].print(f"{timestamp}: {name}")

            # Update GUI
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['-IMAGE-'].update(data=imgbytes)
            window['-COUNT-'].update(f'Face Count: {len(faces)}')
            window['-FPS-'].update(f'FPS: {fps:.2f}')

window.close()
if cap is not None:
    cap.release()