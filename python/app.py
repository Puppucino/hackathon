import os
import logging
import cv2
import numpy as np
import face_recognition
import streamlit as st
from PIL import Image
import tempfile
import google.generativeai as genai
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid

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

# Configure Gemini
API_KEY = 'AIzaSyBTg93wVpgAaKIry1I9GMPaVXE9046jbr8'  # Replace with your actual API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./images.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class LocalImage(Base):
    __tablename__ = "local_images"
    id = Column(String, primary_key=True)
    filename = Column(String)
    local_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

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

def extract_text_from_image(image_path: str) -> dict:
    try:
        if not os.path.exists(image_path):
            logging.error(f"Image file not found at {image_path}")
            return {
                "id_number": "",
                "full_name": "",
                "address": ""
            }
            
        file_size = os.path.getsize(image_path)
        logging.info(f"Processing image of size: {file_size} bytes")

        image = genai.upload_file(
            path=image_path,
            mime_type="image/jpeg"
        )
        
        logging.info("Successfully uploaded image to Gemini")

        prompt = """
        Extract the following information from the ID card.
        Return ONLY the values without any numbering or additional text, in this exact format:

        ID Number: [extracted number]
        Full Name: [extracted name]
        Address: [extracted address]

        Do not include any numbers (1,2,3) or additional text in the response.
        """

        response = model.generate_content(
            contents=[prompt, image],
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "block_none",
                "HARM_CATEGORY_HATE_SPEECH": "block_none",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
            }
        )
        
        # Parse the response and clean up any numbering
        lines = response.text.strip().split('\n')
        cleaned_lines = [line.replace('1. ', '').replace('2. ', '').replace('3. ', '') for line in lines]
        
        extracted_info = {
            "id_number": cleaned_lines[0].replace('ID Number:', '').strip() if len(cleaned_lines) > 0 else "",
            "full_name": cleaned_lines[1].replace('Full Name:', '').strip() if len(cleaned_lines) > 1 else "",
            "address": cleaned_lines[2].replace('Address:', '').strip() if len(cleaned_lines) > 2 else ""
        }
        
        logging.info(f"Extracted info: {extracted_info}")
        return extracted_info

    except Exception as e:
        logging.error(f"Error extracting text: {str(e)}")
        return {
            "id_number": "",
            "full_name": "",
            "address": ""
        }

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

        if st.button("Validate Face Match and Extract Info"):
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

                # Extract and display information
                st.write("### Extracted Information")
                extracted_info = extract_text_from_image(id_image_path)
                
                # Display each field on a new line
                st.write(f"ID Number: {extracted_info['id_number']}")
                st.write(f"Full Name: {extracted_info['full_name']}")
                st.write(f"Address: {extracted_info['address']}")

                # Store in database
                unique_id = str(uuid.uuid4())
                db = SessionLocal()
                db_image = LocalImage(
                    id=unique_id,
                    filename=id_image.name,
                    local_path=id_image_path
                )
                db.add(db_image)
                db.commit()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
