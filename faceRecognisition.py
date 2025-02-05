from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("liveness_model.h5")

def predict_liveness(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Real Face ✅"
    else:
        return "Fake Face (Spoof) ❌"

print(predict_liveness("test_image.jpg"))
