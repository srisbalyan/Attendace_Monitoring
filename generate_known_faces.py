import cv2
import dlib
import numpy as np
import os
import pickle

# Load required dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def generate_encoding(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    if len(faces) != 1:
        print(f"Warning: Found {len(faces)} faces in {image_path}. Skipping.")
        return None
    
    shape = shape_predictor(gray, faces[0])
    face_encoding = np.array(face_recognizer.compute_face_descriptor(img, shape))
    
    return face_encoding

def main():
    known_faces = {}
    images_dir = "known_faces_images"  # Directory containing images of known individuals
    
    for filename in os.listdir(images_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # Split the filename into name and identifier
            name_parts = os.path.splitext(filename)[0].split('_')
            name = '_'.join(name_parts[:-1])  # Join all parts except the last one
            
            image_path = os.path.join(images_dir, filename)
            encoding = generate_encoding(image_path)
            
            if encoding is not None:
                if name not in known_faces:
                    known_faces[name] = []
                known_faces[name].append(encoding)
                print(f"Added encoding for {name} from {filename}")
    
    # Calculate average encoding for each person
    for name, encodings in known_faces.items():
        known_faces[name] = np.mean(encodings, axis=0)
        print(f"Calculated average encoding for {name} from {len(encodings)} images")
    
    # Save encodings to a file
    with open("known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)
    
    print(f"Saved {len(known_faces)} face encodings to known_faces.pkl")

if __name__ == "__main__":
    main()