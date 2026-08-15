import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Set image size
img_size = 128
dataset_path = "dataset"
categories = os.listdir(dataset_path)
data = []
labels = []

# Loop through each category folder
for label, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)
        except:
            print(f"Failed to load image: {image_path}")

X = np.array(data) / 255.0
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total Samples: {len(X)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# Show one sample
plt.imshow(X_train[0])
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()
