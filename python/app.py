from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import face_alignment
from skimage.transform import warp, AffineTransform
import matplotlib.pyplot as plt
import face_recognition
import streamlit as st
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_file(file, filename: str) -> str:
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Handle both FastAPI's UploadFile and Streamlit's UploadedFile
    if hasattr(file, 'file'):  # FastAPI UploadFile
        with open(filepath, "wb") as buffer:
            buffer.write(file.file.read())
    else:  # Streamlit UploadedFile
        with open(filepath, "wb") as buffer:
            buffer.write(file.getbuffer())
            
    logging.info(f"File saved to {filepath}")
    return filepath

def visualize_image(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def extract_and_align_face(image_path: str, title_prefix: str) -> tuple:
    img = cv2.imread(image_path)
    if img is None:
        logging.error("Failed to read image.")
        return None, None

    # Load the image using face_recognition
    face_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(face_image)

    if face_locations:
        logging.info(f"Face locations: {face_locations}")

        # Use the first detected face
        top, right, bottom, left = face_locations[0]
        face_landmarks_list = face_recognition.face_landmarks(face_image, face_locations)

        if face_landmarks_list:
            landmarks = face_landmarks_list[0]
            logging.info(f"Landmarks: {landmarks}")

            # Visualize features
            visualize_image(f"{title_prefix} Features", img)

            # Extract the face region
            face_region = img[top:bottom, left:right]

            # Resize the face region to a consistent size
            aligned_face = cv2.resize(face_region, (112, 112))

            # Denoise the aligned face
            denoised_face = cv2.fastNlMeansDenoisingColored(aligned_face, None, 10, 10, 7, 21)

            # Visualize aligned face
            visualize_image(f"{title_prefix} Aligned Face", denoised_face)

            # Get face encoding
            face_encoding = face_recognition.face_encodings(face_image, [face_locations[0]])[0]

            return face_encoding, denoised_face
        else:
            logging.warning("No landmarks detected.")
            return None, None
    else:
        logging.warning("No face detected.")
        return None, None

def debug_image_loading(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        logging.error("Failed to read image.")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Loaded Image")
        plt.axis('off')
        plt.show()

def compare_faces(control_face_path: str, target_image_path: str) -> tuple[bool, float]:
    try:
        # Debug image loading
        debug_image_loading(control_face_path)
        debug_image_loading(target_image_path)

        # Load and encode the faces
        control_face_image = face_recognition.load_image_file(control_face_path)
        target_face_image = face_recognition.load_image_file(target_image_path)

        control_face_locations = face_recognition.face_locations(control_face_image)
        target_face_locations = face_recognition.face_locations(target_face_image)

        logging.info(f"Control face locations: {control_face_locations}")
        logging.info(f"Target face locations: {target_face_locations}")

        control_face_encodings = face_recognition.face_encodings(control_face_image, control_face_locations)
        target_face_encodings = face_recognition.face_encodings(target_face_image, target_face_locations)

        if not control_face_encodings:
            logging.error("No faces found in the control image.")
            return False, 1.0

        if not target_face_encodings:
            logging.error("No faces found in the target image.")
            return False, 1.0

        control_face_encoding = control_face_encodings[0]
        target_face_encoding = target_face_encodings[0]

        # Compare faces with a custom tolerance
        tolerance = 0.6
        results = face_recognition.compare_faces([control_face_encoding], target_face_encoding, tolerance=tolerance)
        face_distance = face_recognition.face_distance([control_face_encoding], target_face_encoding)[0]

        logging.info(f"Face distance: {face_distance}, Tolerance: {tolerance}")

        is_validated = results[0]

        return bool(is_validated), float(face_distance)
    except Exception as e:
        logging.error(f"An error occurred during face comparison: {e}")
        return False, 1.0

def face_distance_to_confidence(face_distance: float, threshold: float = 0.6) -> float:
    if face_distance > threshold:
        return 0.0
    else:
        return (1.0 - face_distance / threshold) * 100.0

@app.post("/api/validate/")
async def validate_images(id_image: UploadFile = File(...), selfie_image: UploadFile = File(...)):
    try:
        # Save uploaded files
        id_image_path = save_file(id_image, "id_image.jpg")
        selfie_image_path = save_file(selfie_image, "selfie_image.jpg")
        
        # Extract face encodings
        id_face_encoding, _ = extract_and_align_face(id_image_path, "ID Image")
        if id_face_encoding is None:
            logging.error("Failed to extract face from ID image")
            raise HTTPException(status_code=400, detail="Face not found in ID image")
            
        selfie_face_encoding, _ = extract_and_align_face(selfie_image_path, "Selfie Image")
        if selfie_face_encoding is None:
            logging.error("Failed to extract face from selfie image")
            raise HTTPException(status_code=400, detail="Face not found in selfie image")
        
        # Compare faces with a custom tolerance
        tolerance = 0.6
        results = face_recognition.compare_faces([id_face_encoding], selfie_face_encoding, tolerance=tolerance)
        face_distance = face_recognition.face_distance([id_face_encoding], selfie_face_encoding)[0]

        logging.info(f"Face distance: {face_distance}, Tolerance: {tolerance}")

        # Calculate confidence
        confidence = face_distance_to_confidence(face_distance)

        return JSONResponse(content={
            "is_validated": bool(results[0]),
            "face_distance": float(face_distance),
            "confidence": float(confidence)
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Add Streamlit interface
def streamlit_app():
    st.title("Face Validation System")
    st.write("Upload an ID photo and a selfie to verify if they match")

    # File uploaders
    id_image = st.file_uploader("Upload ID Photo", type=['jpg', 'jpeg', 'png'])
    selfie_image = st.file_uploader("Upload Selfie", type=['jpg', 'jpeg', 'png'])

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

# Modify main to handle both FastAPI and Streamlit
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        streamlit_app()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8001)
