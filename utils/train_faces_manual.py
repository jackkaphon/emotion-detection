import pickle
import os
import face_recognition


def train_faces():
    known_faces = {}
    faces_folder = 'uploads/faces'

    for person_name in os.listdir(faces_folder):
        person_folder = os.path.join(faces_folder, person_name)
        if os.path.isdir(person_folder):
            known_faces[person_name] = []

            for file_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, file_name)
                try:
                    face_image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(face_image)[
                        0]
                    known_faces[person_name].append(face_encoding)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    # Save the known_faces dictionary to a file in the root of the project
    known_faces_file = "known_faces_data.pkl"
    with open(known_faces_file, "wb") as f:
        pickle.dump(known_faces, f)

    return f"Training completed. Known faces data saved to {known_faces_file}"


train = train_faces()
print(train)
