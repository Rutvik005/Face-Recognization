
import cv2
import numpy as np
import face_recognition
import os
import glob

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*"))

        if not images_path:
            print("No encoding images found.")
            return

        print(f"{len(images_path)} encoding images found.")

        for img_path in images_path:
            print(f"Loading image: {img_path}")

            img = cv2.imread(img_path)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            face_encodings = face_recognition.face_encodings(rgb_img)

            if face_encodings:
                img_encoding = face_encodings[0]

                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_confidences = []

        if not self.known_face_encodings or not face_encodings:
            return np.array([]), [], []

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.6:  # Adjust the threshold as needed
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
            else:
                name = "Unknown"
                confidence = 0.0

            face_names.append(name)
            face_confidences.append(confidence)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names, face_confidences

sfr = SimpleFacerec()
sfr.load_encoding_images("images")

video_path = "path_to_video.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations, face_names, face_confidences = sfr.detect_known_faces(frame)
    for face_loc, name, confidence in zip(face_locations, face_names, face_confidences):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        print(name, confidence)

        confidence_threshold = 0.5
        if confidence > confidence_threshold:
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        else:
            cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
