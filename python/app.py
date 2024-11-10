import os
import logging
import cv2
import numpy as np
import face_recognition
import streamlit as st
from PIL import Image
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure page
st.set_page_config(
    page_title="Face Validation System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create uploads directory in temp location for Streamlit Cloud
UPLOAD_FOLDER = tempfile.gettempdir()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_file(file, filename: str) -> str:
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buffer:
        buffer.write(file.getbuffer())
    logging.info(f"File saved to {filepath}")
    return filepath

def extract_and_align_face(image_path: str, title_prefix: str) -> tuple:
    img = cv2.imread(image_path)
    if img is None:
        logging.error("Failed to read image.")
        return None, None

    # Load the image using face_recognition
    face_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(face_image)

    if face_locations:
        # Use the first detected face
        top, right, bottom, left = face_locations[0]
        face_landmarks_list = face_recognition.face_landmarks(face_image, face_locations)

        if face_landmarks_list:
            # Extract the face region
            face_region = img[top:bottom, left:right]

            # Resize the face region to a consistent size
            aligned_face = cv2.resize(face_region, (112, 112))

            # Denoise the aligned face
            denoised_face = cv2.fastNlMeansDenoisingColored(aligned_face, None, 10, 10, 7, 21)

            # Get face encoding
            face_encoding = face_recognition.face_encodings(face_image, [face_locations[0]])[0]

            return face_encoding, denoised_face
        else:
            logging.warning("No landmarks detected.")
            return None, None
    else:
        logging.warning("No face detected.")
        return None, None

def compare_faces(control_face_path: str, target_image_path: str) -> tuple[bool, float]:
    try:
        # Load and encode the faces
        control_face_image = face_recognition.load_image_file(control_face_path)
        target_face_image = face_recognition.load_image_file(target_image_path)

        control_face_locations = face_recognition.face_locations(control_face_image)
        target_face_locations = face_recognition.face_locations(target_face_image)

        control_face_encodings = face_recognition.face_encodings(control_face_image, control_face_locations)
        target_face_encodings = face_recognition.face_encodings(target_face_image, target_face_locations)

        if not control_face_encodings or not target_face_encodings:
            return False, 1.0

        control_face_encoding = control_face_encodings[0]
        target_face_encoding = target_face_encodings[0]

        # Compare faces with a custom tolerance
        tolerance = 0.6
        results = face_recognition.compare_faces([control_face_encoding], target_face_encoding, tolerance=tolerance)
        face_distance = face_recognition.face_distance([control_face_encoding], target_face_encoding)[0]

        return bool(results[0]), float(face_distance)
    except Exception as e:
        logging.error(f"An error occurred during face comparison: {e}")
        return False, 1.0

def main():
    st.title("Face Validation System")
    st.write("Upload an ID photo and a selfie to verify if they match")

    # ID Photo uploader
    id_image = st.file_uploader("Upload ID Photo", type=['jpg', 'jpeg', 'png'])

    # Selfie section with both upload and capture options
    st.write("### Selfie Input")
    selfie_method = st.radio("Choose selfie input method:", ["Upload Image", "Capture from Camera"])
    
    selfie_image = None
    if selfie_method == "Upload Image":
        selfie_image = st.file_uploader("Upload Selfie", type=['jpg', 'jpeg', 'png'])
    else:
        camera_image = st.camera_input("Take a selfie")
        if camera_image:
            selfie_image = camera_image

    if id_image and selfie_image:
        # Display uploaded images
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original ID Photo")
            st.image(id_image, use_column_width=True)
        with col2:
            st.write("Original Selfie")
            st.image(selfie_image, use_column_width=True)

        if st.button("Validate Face Match"):
            try:
                # Save uploaded files
                id_image_path = save_file(id_image, "id_image.jpg")
                selfie_image_path = save_file(selfie_image, "selfie_image.jpg")

                # Extract faces and get aligned images
                _, id_aligned_face = extract_and_align_face(id_image_path, "ID Image")
                _, selfie_aligned_face = extract_and_align_face(selfie_image_path, "Selfie Image")

                if id_aligned_face is None or selfie_aligned_face is None:
                    st.error("Could not detect faces in one or both images")
                    return

                # Display aligned faces
                st.write("---")
                st.write("### Aligned Faces")
                col3, col4 = st.columns(2)
                with col3:
                    st.write("Aligned ID Photo")
                    st.image(cv2.cvtColor(id_aligned_face, cv2.COLOR_BGR2RGB), use_column_width=True)
                with col4:
                    st.write("Aligned Selfie")
                    st.image(cv2.cvtColor(selfie_aligned_face, cv2.COLOR_BGR2RGB), use_column_width=True)

                # Compare faces and show result
                is_match, _ = compare_faces(id_image_path, selfie_image_path)
                st.write("### Result")
                st.write(f"Match Status: {'✅ Match' if is_match else '❌ No Match'}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
