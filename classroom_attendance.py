import cv2
import numpy as np
from deepface import DeepFace
import PySimpleGUI as sg
import time
import os
import threading

print("Script started")

# Initialize DeepFace model
print("Initializing DeepFace model...")
model = DeepFace.build_model('Facenet')
print("DeepFace model initialized")

# Load known faces and calculate average encodings
print("Loading known faces...")
known_faces = {}
known_faces_dir = "known_faces_images"
temp_encodings = {}

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # Extract the person's name by removing the unique identifier
        name_parts = os.path.splitext(filename)[0].split('_')
        name = '_'.join(name_parts[:-1])
        
        img_path = os.path.join(known_faces_dir, filename)
        print(f"Processing {img_path}")
        embedding = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=False, detector_backend='opencv')[0]["embedding"]

        if name not in temp_encodings:
            temp_encodings[name] = []
        temp_encodings[name].append(embedding)

# Calculate the average encoding for each person
for name, encodings in temp_encodings.items():
    known_faces[name] = np.mean(encodings, axis=0)
    print(f"Calculated average encoding for {name} from {len(encodings)} images")

print(f"Loaded {len(known_faces)} known faces.")

# Function to recognize face
def recognize_face(face_embedding):
    min_distance = float('inf')
    recognized_name = "Unknown"
    for name, known_embedding in known_faces.items():
        distance = np.linalg.norm(np.array(face_embedding) - np.array(known_embedding))
        if distance < min_distance:
            min_distance = distance
            recognized_name = name
    confidence = 1 - (min_distance / 2)  # Normalize confidence
    return recognized_name, confidence

# Function to determine attentiveness based on emotion
def determine_attentiveness(emotion):
    attentive_emotions = ['neutral', 'happy', 'surprise', 'fear']
    return "Attentive" if emotion in attentive_emotions else "Not Attentive"

# Function to analyze faces using DeepFace
def analyze_faces(frame, results):
    try:
        resized_frame = cv2.resize(frame, (640, 480))
        analyses = DeepFace.analyze(
            img_path=resized_frame,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        results.append(analyses)
    except Exception as e:
        print(f"An error occurred during face analysis: {e}")
        results.append(None)

# GUI Layout
print("Setting up GUI...")
layout = [
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Text('Face Count: 0', key='-COUNT-'), sg.Text('FPS: 0.00', key='-FPS-')],
    [sg.Button('Start'), sg.Button('Stop'), sg.Button('Exit')]
]

window = sg.Window('Classroom Attendance System', layout, finalize=True)
print("GUI setup complete")

# Initialize variables
cap = None
prev_time = 0
frame_count = 0
ret = False
face_data = []

# Function to capture frames continuously
def capture_frames():
    global cap, frame, ret
    while cap is not None:
        ret, frame = cap.read()

thread = None

print("Entering main loop")
while True:
    event, values = window.read(timeout=20)
    if event in (sg.WINDOW_CLOSED, 'Exit'):
        print("Exit event received")
        break

    if event == 'Start' and cap is None:
        print("Start button pressed")
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        if not cap.isOpened():
            print("Error: Could not open webcam")
            sg.popup("Error: Could not open webcam")
            cap = None
        else:
            thread = threading.Thread(target=capture_frames)
            thread.start()

    if event == 'Stop' and cap is not None:
        print("Stop button pressed")
        cap.release()
        cap = None
        if thread is not None:
            thread.join()

    if cap is not None and ret:
        try:
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Perform face analysis every 30 frames or if no faces detected yet
            if frame_count % 30 == 0 or not face_data:
                results = []
                analysis_thread = threading.Thread(target=analyze_faces, args=(frame, results))
                analysis_thread.start()
                analysis_thread.join()
                analyses = results[0]

                if analyses:
                    if isinstance(analyses, dict):
                        analyses = [analyses]

                    face_data = []
                    for face in analyses:
                        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                        
                        face_img = frame[y:y+h, x:x+w]
                        embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False, detector_backend='opencv')[0]["embedding"]
                        name, confidence = recognize_face(embedding)

                        face_data.append({
                            'bbox': (x, y, w, h),
                            'name': name,
                            'confidence': confidence,
                            'emotion': face['dominant_emotion'],
                            'age': face['age'],
                            'gender': face['gender'],
                            'attentiveness': determine_attentiveness(face['dominant_emotion'])
                        })

            # Draw bounding boxes and information for all detected faces
            for face in face_data:
                x, y, w, h = face['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                text_lines = [
                    f"Name: {face['name']} ({face['confidence']:.2f})",
                    f"Emotion: {face['emotion']}",
                    f"Age: {face['age']} yrs",
                    f"Gender: {face['gender']}",
                    f"Attentiveness: {face['attentiveness']}"
                ]

                font_scale = 0.5
                font_color = (0, 0, 0)  # Changed to black as requested
                font_thickness = 1  # Control font thickness for better visibility
                line_spacing = 10  # Controls line spacing

                text_x = x + w + 10
                for i, line in enumerate(text_lines):
                    text_y = y + i * line_spacing + 20
                    # Draw a white background for better text visibility
                    (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(frame, (text_x - 2, text_y - text_height - 2), (text_x + text_width + 2, text_y + 2), (255, 255, 255), -1)
                    cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # Update GUI
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['-IMAGE-'].update(data=imgbytes)
            window['-COUNT-'].update(f'Face Count: {len(face_data)}')
            window['-FPS-'].update(f'FPS: {fps:.2f}')
        except Exception as e:
            print(f"Error processing frame: {str(e)}")

print("Main loop exited")
window.close()
if cap is not None:
    cap.release()
print("Script ended")