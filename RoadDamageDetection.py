import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import streamlit as st
from PIL import Image
import tempfile

# Load class labels from the file (directly within the code)
with open("itemName.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Streamlit UI Setup
st.title("YOLO Object Detection on Video")
st.sidebar.header("Upload Video")
uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["Custom Model", "Basic Model Yolo11n"])
skip_frames = st.sidebar.slider("Skip Frames (Processing Frequency)", 1, 5, 3)

if uploaded_video:
    # Load the YOLO model
    if model_choice == "Custom Model":
        model = YOLO('Model/best.pt')
    else:
        model = YOLO('Model/yolo11n.pt')

    st.sidebar.success("Class Labels Loaded Successfully!")

    # Save video temporarily
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())

    # Open the video
    cap = cv2.VideoCapture(temp_video.name)

    # Video display in Streamlit
    stframe = st.empty()
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video processing completed!")
            break

        count += 1
        if count % skip_frames != 0:
            continue

        # Resize the frame
        frame = cv2.resize(frame, (1020, 600))

        # Predict using YOLO
        results = model.predict(frame)
        if len(results[0].boxes.data) == 0:
            continue

        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        # Annotate detections
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d] if d < len(class_list) else "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        # Display the frame in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()

else:
    st.info("Please upload a video file to proceed.")
