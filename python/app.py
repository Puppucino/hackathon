from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from deepface import DeepFace
import cv2
import pytesseract

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_file(file: UploadFile, filename: str) -> str:
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())
    return filepath

def extract_face_from_id(image_path: str) -> str:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use a pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        logging.error("Face not found in ID image.")
        return None
    
    # Assume the first detected face is the one we want
    (x, y, w, h) = faces[0]
    face_image = image[y:y+h, x:x+w]
    face_path = os.path.join(UPLOAD_FOLDER, 'extracted_face.jpg')
    cv2.imwrite(face_path, face_image)
    return face_path

def compare_faces(id_image_path: str, selfie_image_path: str) -> tuple[bool, float]:
    try:
        face_path = extract_face_from_id(id_image_path)
        if not face_path:
            return False, 0.0
        
        # Try a different model and adjust the threshold
        result = DeepFace.verify(img1_path=face_path, img2_path=selfie_image_path, model_name='VGG-Face')
        is_validated = result['verified'] and result['distance'] < 0.5  # Adjusted threshold
        similarity_score = result['distance']
        
        return is_validated, similarity_score
    except Exception as e:
        logging.error(f"An error occurred during face comparison: {e}")
        return False, 0.0

def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(equalized, None, 30, 7, 21)
    
    # Convert back to BGR
    enhanced_image = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    return enhanced_image

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    filepath = save_file(file, 'id_image.jpg')
    return JSONResponse(content={"message": "Image uploaded successfully", "filepath": filepath})

@app.post("/compare")
async def compare_images_endpoint(selfie: UploadFile = File(...)):
    try:
        if not selfie.filename:
            raise HTTPException(status_code=400, detail="No selected selfie")

        selfie_path = save_file(selfie, 'selfie.jpg')
        id_image_path = os.path.join(UPLOAD_FOLDER, 'id_image.jpg')
        is_validated, similarity_score = compare_faces(id_image_path, selfie_path)

        return JSONResponse(content={"is_validated": is_validated, "similarity_score": similarity_score})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)