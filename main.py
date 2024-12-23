import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import streamlit as st
import tempfile
import numpy as np

# Load class labels from the file (directly within the code)
with open("itemName.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Road Line Detection Helper Functions
def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    cropped_img = cv2.bitwise_and(image, mask)
    return cropped_img

def draw_lines(image, hough_lines):
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def process_road_lines(img):
    height = img.shape[0]
    width = img.shape[1]
    roi_vertices = [
        (0, int(height * 0.9)),
        (int(width * 2 / 3), int(height * 0.6)),
        (width, int(height * 0.9))
    ]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.dilate(gray_img, kernel=np.ones((3, 3), np.uint8))
    canny = cv2.Canny(gray_img, 130, 220)
    roi_img = roi(canny, np.array([roi_vertices], np.int32))
    lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, threshold=10, minLineLength=15, maxLineGap=2)
    final_img = draw_lines(img, lines)
    return final_img

# Streamlit UI Setup
st.title("Road Damage and Line Detection and Road Analysis")
menu = st.sidebar.selectbox("Choose Analysis Pipeline", ["Road Damage Detection", "Road Line Detection"])

# Common UI Elements
uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
skip_frames = st.sidebar.slider("Skip Frames (Processing Frequency)", 1, 5, 3)

if menu == "Road Damage Detection":
    st.sidebar.header("Road Damage Detection Settings")
    model_choice = st.sidebar.selectbox("Choose Model", ["Custom Model", "Basic Model Yolo11n"])

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

elif menu == "Road Line Detection":
    st.sidebar.header("Road Line Detection Settings")

    if uploaded_video:
        # Save video temporarily
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())

        # Open the video
        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video processing completed!")
                break

            # Process the frame for road lines
            frame = process_road_lines(frame)

            # Display the frame in Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_column_width=True)

        cap.release()

    else:
        st.info("Please upload a video file to proceed.")
