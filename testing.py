from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
API_KEY = 'AIzaSyDSlRGvF5vFugSrRHc_bd0AW1_GPMl6_1A'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./images.db"  # Using SQLite for simplicity
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

@app.post("/process-local-image/")
async def process_local_image(file: UploadFile = File(...)):
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploaded_images"
        os.makedirs(upload_dir, exist_ok=True)

        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_id = str(uuid.uuid4())
        unique_filename = f"{unique_id}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)

        # Save file locally
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        # Store in database
        db = SessionLocal()
        db_image = LocalImage(
            id=unique_id,
            filename=file.filename,
            local_path=file_path
        )
        db.add(db_image)
        db.commit()

        # Process with Gemini
        with open(file_path, "rb") as img_file:
            image = genai.upload_file(path=file_path, mime_type="image/jpeg")
            prompt = "Extract the text in the image verbatim with the json format {id_number: str, full_name: str, address: str}"
            response = model.generate_content([prompt, image])

        return {
            "id": unique_id,
            "filename": file.filename,
            "local_path": file_path,
            "extracted_text": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/process-local-image/")
# async def process_local_image(file: UploadFile = File(...)):
#     try:
#         # Create uploads directory if it doesn't exist
#         upload_dir = "uploaded_images"
#         os.makedirs(upload_dir, exist_ok=True)

#         # Generate unique filename
#         file_extension = os.path.splitext(file.filename)[1]
#         unique_id = str(uuid.uuid4())
#         unique_filename = f"{unique_id}{file_extension}"
#         file_path = os.path.join(upload_dir, unique_filename)

#         # Process with Gemini
#         with open(file_path, "rb") as img_file:
#             image = genai.upload_file(path=file_path, mime_type="image/jpeg")
#             # Modified prompt to be more specific
#             prompt = """
#             Extract text from the image and return ONLY a clean JSON object with this exact format:
#             {
#                 "id_number": "extracted id number",
#                 "full_name": "extracted full name",
#                 "address": "extracted address"
#             }
#             Do not include markdown formatting, code blocks, or any other text.
#             """
#             response = model.generate_content([prompt, image])
            
#             # Clean up the response
#             text = response.text
#             # Remove markdown code blocks if present
#             text = text.replace('```json', '').replace('```', '').strip()
            
#             # Parse the JSON string to ensure it's valid JSON
#             import json
#             try:
#                 extracted_data = json.loads(text)
#             except json.JSONDecodeError:
#                 # If JSON parsing fails, create a structured response
#                 extracted_data = {
#                     "id_number": "",
#                     "full_name": "",
#                     "address": ""
#                 }

#         return {
#             "id": unique_id,
#             "filename": file.filename,
#             "local_path": file_path,
#             "extracted_text": extracted_data  # Now returns clean JSON object
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Optional: Endpoint to get processed image details
@app.get("/images/{image_id}")
async def get_image_details(image_id: str):
    db = SessionLocal()
    image = db.query(LocalImage).filter(LocalImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return {
        "id": image.id,
        "filename": image.filename,
        "local_path": image.local_path,
        "created_at": image.created_at
    }

# Optional: Cleanup function (you might want to run this periodically)
def cleanup_old_images():
    db = SessionLocal()
    current_time = datetime.utcnow()
    old_images = db.query(LocalImage).filter(
        LocalImage.created_at < current_time - timedelta(days=1)
    ).all()
    
    for image in old_images:
        if os.path.exists(image.local_path):
            os.remove(image.local_path)
        db.delete(image)
    db.commit()