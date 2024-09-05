import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import torch
import tensorflow as tf

# Create the main application window
root = tk.Tk()
root.title("Exercise Detection GUI")
root.geometry("1200x1000")  # Adjust the window size as needed


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load exercise classification model
model1 = tf.keras.models.load_model("PhysioNet.h5")

# Define COCO classes
COCO_CLASSES = ['person']

# Define exercise labels
labels = ['Butterfly', 'Calf raises', 'goddess', 'Hand raises', 'Knee pushups', 'Lowerback strecth',
          'Shoulder press', 'shoulder stretch', 'situps', 'tree', 'wallChair', 'Warmup']

# Load and display background image
background_image = tk.PhotoImage(file="bg.png")
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Cover the entire window

def start_video_capture():
    cap = cv2.VideoCapture(0)  # Open the default camera (usually the built-in webcam)
    display_video(cap)

def upload_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        display_video(cap)

# Function to display video and perform detection
def display_video(cap):
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.resize(frame, (224, 224))

        results = np.argmax(model1.predict(np.array([image])))

        detections = model(frame)

        person_detections = detections.pred[0][detections.pred[0][:, 5] == 0]

        for det in person_detections:
            x1, y1, x2, y2, conf, _ = det.tolist()
            label = f'Score: {conf:.2f} - Exercise: {labels[results]}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
title_label = tk.Label(root, text="KALASALINGAM ACADEMY OF RESEARCH AND EDUCATION", font=("Roboto", 39))

# Place the title label at the top of the window
title_label.grid(row=0, column=0, columnspan=5, padx=10, pady=20)
# Create buttons for video options
webcam_button = tk.Button(root, text="Capture from Webcam", command=start_video_capture,width=20, height=2)
file_button = tk.Button(root, text="Upload Video File", command=upload_video_file,width=20, height=2)

clg = tk.PhotoImage(file="clg.png")
clg1 = tk.Label(root, image=clg)
clg1.grid(row=3, column=2, padx=10, pady=10)

webcam_button.grid(row=1, column=2, padx=10, pady=50)
file_button.grid(row=2, column=2, padx=10, pady=50)

# Add student images inline at the bottom of the page using the grid manager
student_image1 = tk.PhotoImage(file="s1.png")
student_image2 = tk.PhotoImage(file="s2.png")
student_image3 = tk.PhotoImage(file="s3.png")
student_image4 = tk.PhotoImage(file="s4.png")
student_image5 = tk.PhotoImage(file="s5.png")

student_label1 = tk.Label(root, image=student_image1)
student_label2 = tk.Label(root, image=student_image2)
student_label3 = tk.Label(root, image=student_image3)
student_label4 = tk.Label(root, image=student_image4)
student_label5 = tk.Label(root, image=student_image5)

student_label1.grid(row=5, column=0, padx=10, pady=10)
student_label2.grid(row=5, column=1, padx=10, pady=10)
student_label3.grid(row=5, column=2, padx=10, pady=10)
student_label4.grid(row=5, column=3, padx=10, pady=10)
student_label5.grid(row=5, column=4, padx=10, pady=10)
# Start the Tkinter main loop
root.mainloop()