import sys
sys.path.append("D:\\python_libs")

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

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
# 2. Convert to arrays
# -----------------------------
data = np.array(data)
labels = np.array(labels)

# -----------------------------
# 3. Shuffle (IMPORTANT)
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
# 5. One-hot encoding
# -----------------------------
labels_cat = to_categorical(labels, 2)

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_cat, test_size=0.2, stratify=labels
)

# -----------------------------
# 7. Data Augmentation (SAFE)
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# -----------------------------
# 8. SIMPLE CNN (BEST FOR YOUR DATA)
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
# 10. Early Stopping
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# 11. Train
# -----------------------------
model.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    epochs=15,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -----------------------------
# 12. Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("✅ Test Accuracy:", acc)

# -----------------------------
# 13. Save Model
# -----------------------------
model.save("model.keras")

print("✅ Training complete. Model saved as model.keras")