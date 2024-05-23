from flask import Flask, jsonify
from flask_cors import CORS
from image_routes import image_bp
from flask_socketio import SocketIO, emit

import os
import pickle
import cv2
import numpy as np
import face_recognition
from io import BytesIO
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

app.register_blueprint(image_bp, url_prefix='/images')

# Load the known_faces dictionary from the file
with open("known_faces_data.pkl", "rb") as f:
    known_faces = pickle.load(f)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello():
    return 'Hello, World!'

# simple websocket example
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# Load the pre-trained FER model
emotion_model = load_model("models/FER_model.h5")

# Define the emotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Use the ThreadPoolExecutor to run the face recognition in a separate thread
executor = ThreadPoolExecutor(max_workers=1)


@socketio.on('recognize')
def recognize_image(data):
    try:
        # Read the image file from the request
        image_data = data['image']

        # Convert the image data to a numpy array
        nparr = np.frombuffer(image_data, np.uint8)

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

        emit('recognized_faces', recognized_faces)
    except Exception as e:
        emit('error', {'error': str(e)})


# Custom 500 Internal Server Error handler
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True)
