# Classroom Attendance System

This project implements an automated classroom attendance system using face recognition.

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment
4. Install dependencies: `pip install -r requirements.txt`
5. Download required dlib models (see README for details)
6. Run the script: `python classroom_attendance.py`

## Requirements

- Python 3.7+
- See requirements.txt for Python package dependencies
- dlib face recognition models (not included in repo due to size)

## Usage

[Add usage instructions here]

## License

[Add license information here]


## Dlib Models

This project requires two dlib models which are not included in the repository due to their size:

1. dlib_face_recognition_resnet_model_v1.dat
2. shape_predictor_68_face_landmarks.dat

To download these models:

1. Visit the dlib model download page: http://dlib.net/files/
2. Download the following files:
   - shape_predictor_68_face_landmarks.dat.bz2
   - dlib_face_recognition_resnet_model_v1.dat.bz2
3. Extract the .bz2 files to get the .dat files
4. Place both .dat files in the project root directory

Note: These models are essential for face recognition and landmark detection. The project will not run without them.


## Using Webcam

By default, the script is set to use your computer's webcam. If you want to use a different camera or switch back to an RTSP stream:

1. Open `classroom_attendance.py`
2. Locate the line `cap = cv2.VideoCapture(0)`
3. Change `0` to the index of your desired camera (e.g., `1` for a second webcam)
4. For RTSP stream, replace the line with:
   ```python
   cap = cv2.VideoCapture("rtsp://your_rtsp_stream_url_here")


## Preparing Known Faces

To use face recognition with multiple images per person:

1. Create a directory named `known_faces_images` in the project root.
2. Add clear, front-facing images of known individuals to this directory.
   Name each image file as follows: "person_name_uniqueidentifier.jpg"
   (e.g., "john_doe_1.jpg", "john_doe_2.jpg", "jane_smith_1.jpg")
3. Run the face encoding generator:python generate_known_faces.py
4. This will create `known_faces.pkl`, which contains average encodings for each person.

Note: 
- Ensure you have permission to use individuals' images for this purpose.
- Using multiple images per person can improve recognition accuracy.
- The script will calculate an average encoding for each person from their multiple images.

## Environment Setup

This project is developed using Python 3.12.2. To set up the environment:

1. Ensure you have Python 3.12.2 installed
2. Create a virtual environment:python -m venv venv
3. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- MacOS/Linux: `source venv/bin/activate`
4. Install the required packages:pip install -r requirements.txt
Note: If you encounter issues with dlib installation, refer to the Troubleshooting section below.

## Troubleshooting

If you encounter issues installing dlib:
1. Ensure you have CMake installed: `pip install cmake`, 
2. Install Visual Studio Build Tools 2019 or later with "Desktop development with C++" workload, then
2a. Try installing dlib:
3. If issues persist, download a pre-built wheel from [Dlib-Wheels](https://github.com/sachadee/Dlib-Wheels) and install it manually:
4. For RetinaFace, you might need to install it from the GitHub repository (pip install git+https://github.com/serengil/retinaface.git)

## Recent Updates
- Fixed issues with OpenCV tracking methods
- Improved face detection using RetinaFace
- Simplified the main script for better performance
- Added FPS counter to the GUI