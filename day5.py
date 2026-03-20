import sys
sys.path.append("D:\\python_libs")

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")

# Load test image
img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
    exit()

# Preprocess image
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = np.reshape(img, (1, 64, 64, 3))

# Predict
prediction = model.predict(img)

# Output
if prediction[0][0] > prediction[0][1]:
    print("🌱 Prediction: Crop")
else:
    print("🌿 Prediction: Weed")