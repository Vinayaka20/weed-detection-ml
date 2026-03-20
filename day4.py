import sys
sys.path.append("D:\\python_libs")   # TensorFlow path

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = []
labels = []

dataset_path = "dataset"

for label, category in enumerate(["crop", "weed"]):
    folder_path = os.path.join(dataset_path, category)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))
        data.append(img)
        labels.append(label)

# -----------------------------
# 2. Convert to NumPy arrays
# -----------------------------
data = np.array(data)
labels = np.array(labels)

# -----------------------------
# 3. Shuffle data
# -----------------------------
indices = np.arange(len(data))
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

# -----------------------------
# 4. Normalize
# -----------------------------
data = data / 255.0

# -----------------------------
# 5. Convert labels
# -----------------------------
labels = to_categorical(labels, 2)

# -----------------------------
# 6. Dataset check
# -----------------------------
print("Crop count:", np.sum(labels[:,0]))
print("Weed count:", np.sum(labels[:,1]))

# -----------------------------
# 7. Data Augmentation
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(data)

# -----------------------------
# 8. CNN Model + Dropout
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(2, activation='softmax')
])

# -----------------------------
# 9. Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 10. Train (FIXED - no validation_split)
# -----------------------------
model.fit(
    datagen.flow(data, labels, batch_size=8),
    epochs=15
)

# -----------------------------
# 11. Save Model
# -----------------------------
model.save("model.h5")

print("✅ Training complete. Model saved as model.h5")