import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained liveness detection model (Assumes you have a trained model)
model = load_model("liveness_model.h5")  # Replace with your trained model

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (32, 32))  # Resize for model input
        face = np.expand_dims(face, axis=0) / 255.0  # Normalize

        # Predict Real or Fake
        prediction = model.predict(face)
        label = "Real" if prediction > 0.5 else "Fake"

        # Draw Result on Frame
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
