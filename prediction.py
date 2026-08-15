import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk

# Load the trained model
model = load_model("leaf_disease_model.keras")
print("Model loaded successfully!")
# Classes in your dataset
classes = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']

# Function to select and predict image
def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # Load image
    img = cv2.imread(file_path)
    if img is None:
        result_label.config(text="Error: Image not loaded.")
        return
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    
    # Predict
    pred = model.predict(img_resized)
    predicted_class = classes[np.argmax(pred)]
    
    # Show prediction
    result_label.config(text=f"Prediction: {predicted_class}")
    
    # Display image in GUI
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil.resize((250, 250)))
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Create GUI
root = tk.Tk()
root.title("Leaf Disease Prediction")

Button(root, text="Select Leaf Image", command=select_image).pack(pady=10)
image_label = Label(root)
image_label.pack()
result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
