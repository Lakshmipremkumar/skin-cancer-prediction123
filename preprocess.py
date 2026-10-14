import pandas as pd
import os
import cv2

# Load metadata
data = pd.read_csv("dataset/HAM10000_metadata.csv")

images = []
labels = []

for index, row in data.iterrows():
    image_id = row["image_id"]

    path1 = os.path.join("dataset", "HAM10000_images_part_1", image_id + ".jpg")
    path2 = os.path.join("dataset", "HAM10000_images_part_2", image_id + ".jpg")

    image_path = None
    if os.path.exists(path1):
        image_path = path1
    elif os.path.exists(path2):
        image_path = path2

    if image_path:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        images.append(image)
        labels.append(row["dx"])

print("Total Images:", len(images))
print("Total Labels:", len(labels))

from sklearn.preprocessing import LabelEncoder

# Convert labels into numbers
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

print("\nDisease Classes:")
print(encoder.classes_)

print("\nFirst 10 Encoded Labels:")
print(encoded_labels[:10])
from sklearn.model_selection import train_test_split
import numpy as np

images = np.array(images)
encoded_labels = np.array(encoded_labels)

X_train, X_test, y_train, y_test = train_test_split(
    images,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)

print("Training Images:", X_train.shape)
print("Testing Images:", X_test.shape)

# Save the dataset
np.save("model/X_train.npy", X_train)
np.save("model/X_test.npy", X_test)
np.save("model/y_train.npy", y_train)
np.save("model/y_test.npy", y_test)

print("Dataset saved successfully!")

from sklearn.model_selection import train_test_split
import numpy as np

images = np.array(images)
encoded_labels = np.array(encoded_labels)

X_train, X_test, y_train, y_test = train_test_split(
    images,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)

print("Training Images:", X_train.shape)
print("Testing Images:", X_test.shape)

# Save dataset
np.save("model/X_train.npy", X_train)
np.save("model/X_test.npy", X_test)
np.save("model/y_train.npy", y_train)
np.save("model/y_test.npy", y_test)

print("Dataset saved successfully!")