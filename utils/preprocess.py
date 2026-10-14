import pandas as pd
import os
import cv2
import numpy as np

# Load metadata
data = pd.read_csv("dataset/HAM10000_metadata.csv")

images = []
labels = []

# Read images
for index, row in data.iterrows():

    image_id = row["image_id"]

    path1 = os.path.join("dataset", "HAM10000_images_part_1", image_id + ".jpg")
    path2 = os.path.join("dataset", "HAM10000_images_part_2", image_id + ".jpg")

    if os.path.exists(path1):
        image_path = path1
    elif os.path.exists(path2):
        image_path = path2
    else:
        continue

    # Read image
    image = cv2.imread(image_path)

    # Resize image
    image = cv2.resize(image, (224, 224))

    # Normalize image
    image = image / 255.0

    images.append(image)
    labels.append(row["dx"])

print("Total Images Loaded:", len(images))
print("Total Labels:", len(labels))