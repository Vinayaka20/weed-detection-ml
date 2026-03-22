import sys
sys.path.append("D:\\python_libs")

import cv2
import numpy as np
import os
import tensorflow as tf

# -----------------------------
# 1. Load Model
# -----------------------------
model = tf.keras.models.load_model("model.keras")

# -----------------------------
# 2. Test Images Folder
# -----------------------------
test_path = "test_images"

# -----------------------------
# 3. Loop through images
# -----------------------------
for img_name in os.listdir(test_path):

    img_path = os.path.join(test_path, img_name)

    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Could not load {img_name}")
        continue

    # -----------------------------
    # 4. Preprocess
    # -----------------------------
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # -----------------------------
    # 5. Prediction
    # -----------------------------
    prediction = model.predict(img_input)[0]

    crop_prob = prediction[0]
    weed_prob = prediction[1]

    confidence = max(crop_prob, weed_prob)

    # -----------------------------
    # 6. Output
    # -----------------------------
    if confidence < 0.6:
        print(f"{img_name} → ⚠️ Uncertain ({confidence*100:.2f}%)")
    elif crop_prob > weed_prob:
        print(f"{img_name} → 🌾 Crop ({crop_prob*100:.2f}%)")
    else:
        print(f"{img_name} → 🌿 Weed ({weed_prob*100:.2f}%)")

    # -----------------------------
    # 7. Show image (optional)
    # -----------------------------
    cv2.imshow("Image", img)
    cv2.waitKey(1000)

cv2.destroyAllWindows()