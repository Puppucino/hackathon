import streamlit as st
from PIL import Image
import io
import cv2
import time
import face_recognition  # Ensure you have this library installed

# Initialize session state
if 'id_image' not in st.session_state:
    st.session_state.id_image = None
if 'selfie_image' not in st.session_state:
    st.session_state.selfie_image = None

st.title("User Signup")
username = st.text_input("Enter a unique username")
id_number = st.text_input("Enter your ID number")
id_address = st.text_input("Enter your ID address")
name = st.text_input("Enter your name")

# Image uploader for ID card
uploaded_file = st.file_uploader("Upload your ID card image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')  # Convert RGBA to RGB
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    st.session_state.id_image = img_bytes.getvalue()
    st.image(st.session_state.id_image, caption="ID Card Image", use_column_width=True)

# Prompt for liveness selfie
if st.session_state.id_image and st.button("Take Liveness Selfie"):
    try:
        cam = cv2.VideoCapture(0)
        st.write("Camera opened. Press 'Capture' to take a selfie.")
        placeholder = st.empty()
        capture_button = st.button("Capture Selfie")
        while True:
            ret, frame = cam.read()
            if not ret:
                raise Exception("Failed to grab frame")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            placeholder.image(img)
            if capture_button:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                st.session_state.selfie_image = img_bytes.getvalue()
                cam.release()
                break
            time.sleep(0.1)
    except Exception as e:
        st.error(f"Error using camera: {e}")

# Compare ID image and selfie
if st.session_state.id_image and st.session_state.selfie_image:
    id_image = face_recognition.load_image_file(io.BytesIO(st.session_state.id_image))
    selfie_image = face_recognition.load_image_file(io.BytesIO(st.session_state.selfie_image))

    id_encodings = face_recognition.face_encodings(id_image)
    selfie_encodings = face_recognition.face_encodings(selfie_image)

    if id_encodings and selfie_encodings:
        match_results = face_recognition.compare_faces([id_encodings[0]], selfie_encodings[0], tolerance=0.1)
        if match_results[0]:
            st.success("User validated successfully!")
        else:
            st.error("Face does not match. Please try again.")
    else:
        st.error("Could not detect face in one of the images. Please ensure your face is clearly visible.")
