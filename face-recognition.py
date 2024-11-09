import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from PIL import Image
import io

# Define the database URL (SQLite in this case)
DATABASE_URL = 'sqlite:///users.db'

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a base class for the declarative model
Base = declarative_base()

# Define the User model
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    id_image = Column(LargeBinary, nullable=False)
    id_number = Column(String, nullable=False)
    id_address = Column(String, nullable=False)
    name = Column(String, nullable=False)

# Create the database and the user table
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Streamlit app
st.title("User  Signup")

# Username input
username = st.text_input("Enter a unique username")

# Upload ID image
uploaded_id_image = st.file_uploader("Upload your ID image", type=["jpg", "jpeg", "png"])

# Input fields for ID number, ID address, and name
id_number = st.text_input("Enter your ID number")
id_address = st.text_input("Enter your ID address")
name = st.text_input("Enter your name")

# Submit button
if st.button("Sign Up"):
    if username and uploaded_id_image and id_number and id_address and name:
        # Read the image file
        id_image = uploaded_id_image.read()  # Read the file as binary

        # Create a new user instance
        new_user = User(
            username=username,
            id_image=id_image,
            id_number=id_number,
            id_address=id_address,
            name=name
        )

        # Add the new user to the session and commit
        try:
            session.add(new_user)
            session.commit()
            st.success("User  registered successfully!")
        except Exception as e:
            session.rollback()  # Rollback in case of error
            if "UNIQUE constraint failed" in str(e):
                st.error("Username already exists. Please choose a different username.")
            else:
                st.error("An error occurred while registering the user.")
    else:
        st.error("Please fill in all fields and upload your ID image.")

# Close the session when done
session.close()