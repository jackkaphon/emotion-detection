from flask import Blueprint, request, jsonify
from utils.train_faces import train_faces
import os
import pickle
import cv2
import numpy as np
import face_recognition
from io import BytesIO
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor

image_bp = Blueprint('image', __name__)

# Load the known_faces dictionary from the file
with open("known_faces_data.pkl", "rb") as f:
    known_faces = pickle.load(f)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load the pre-trained FER model
emotion_model = load_model("models/FER_model.h5")

# Define the emotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


@image_bp.route('/train', methods=['POST'])
def train_image():
    try:
        data = request.form.to_dict()

        file = request.files['file']

        if 'file' not in request.files:
            print('No file part')
            return errorResponse('No file part', 400)

        if file.filename == '':
            print('No selected file')
            return errorResponse('No selected file', 400)

        if file and allowed_file(file.filename):
            filename = file.filename
            person_name = data.get('name', 'unknown')
            person_folder = os.path.join('uploads/faces', person_name)

            # Create the person's folder if it doesn't exist
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

            existing_images = os.listdir(person_folder)
            image_number = len(existing_images) + 1
            new_image_name = f"{image_number}.jpg"
            file.save(os.path.join(person_folder, new_image_name))
        else:
            return errorResponse('Invalid file type', 400)

        result = train_faces()
        return 'TRAINING COMPLETED'
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@image_bp.route('/recognize', methods=['POST'])
def recognize_image():
    try:
        # Read the image file from the request
        image_data = request.files['file']

        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        if image_data.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if image_data and allowed_file(image_data.filename):
            # Convert the image data to a numpy array
            nparr = np.frombuffer(image_data.read(), np.uint8)

            # Decode the image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Use face recognition on the decoded image
            unknown_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            unknown_face_locations = face_recognition.face_locations(
                unknown_image)
            unknown_face_encodings = face_recognition.face_encodings(
                unknown_image, unknown_face_locations)

            # Initialize list to store recognized faces
            recognized_faces = []

            # Process the image, draw rectangles, etc.
            for face_encoding, face_location in zip(unknown_face_encodings, unknown_face_locations):
                name = "Unknown"
                for known_name, known_encodings in known_faces.items():
                    matches = face_recognition.compare_faces(
                        known_encodings, face_encoding)
                    if True in matches:
                        name = known_name
                        break

                # Emotion detection
                face_img = unknown_image[face_location[0]:face_location[2],
                                         face_location[3]:face_location[1]]
                face_img = cv2.resize(face_img, (48, 48))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_img = np.reshape(
                    face_img, [1, face_img.shape[0], face_img.shape[1], 1])
                face_img = face_img.astype('float32') / 255.0

                emotion_prediction = emotion_model.predict(face_img)[0]
                emotion_label = np.argmax(emotion_prediction)
                emotion = EMOTIONS[emotion_label]

                # Add face location and name to recognized_faces list
                recognized_faces.append(
                    {"name": name, "location": face_location, "emotion": emotion})

        return jsonify(recognized_faces)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
