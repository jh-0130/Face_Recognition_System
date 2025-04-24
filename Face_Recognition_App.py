# Face Recognition Attendance System - Streamlit App
import streamlit as st
import cv2
import numpy as np
import face_recognition
import pickle
import os
from tensorflow.keras.models import load_model
import pandas as pd
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👤",
    layout="wide"
)

# Title and description
st.title("Face Recognition Attendance System")
st.markdown("""
This application uses CNN to recognize faces and mark attendance.
Upload an image containing faces, and the system will identify the individuals and record their attendance.
""")

# Function to load the model and mappings
@st.cache_resource
def load_recognition_model():
    # Update these paths to your model and mapping files
    model_path = "face_recognition_cnn_model.h5"
    label_mapping_path = "label_mapping.pkl"
    reverse_mapping_path = "reverse_mapping.pkl"
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please make sure to upload it.")
        return None, None, None
        
    if not os.path.exists(label_mapping_path) or not os.path.exists(reverse_mapping_path):
        st.error("Label mapping files not found. Please upload them.")
        return None, None, None
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Load label mappings
        with open(label_mapping_path, 'rb') as f:
            label_mapping = pickle.load(f)
        
        with open(reverse_mapping_path, 'rb') as f:
            reverse_mapping = pickle.load(f)
            
        return model, label_mapping, reverse_mapping
    
    except Exception as e:
        st.error(f"Error loading model or mappings: {e}")
        return None, None, None

# Sidebar - Model File Upload
with st.sidebar:
    st.header("Model Files")
    st.markdown("If you don't have the model files in the current directory, upload them here:")
    
    model_file = st.file_uploader("Upload CNN Model (.h5)", type=["h5"])
    label_mapping_file = st.file_uploader("Upload Label Mapping File (.pkl)", type=["pkl"])
    reverse_mapping_file = st.file_uploader("Upload Reverse Mapping File (.pkl)", type=["pkl"])
    
    if model_file and label_mapping_file and reverse_mapping_file:
        # Save uploaded files
        with open("face_recognition_cnn_model.h5", "wb") as f:
            f.write(model_file.getbuffer())
        with open("label_mapping.pkl", "wb") as f:
            f.write(label_mapping_file.getbuffer())
        with open("reverse_mapping.pkl", "wb") as f:
            f.write(reverse_mapping_file.getbuffer())
        st.success("Model files uploaded successfully!")

# Load model and mappings
model, label_mapping, reverse_mapping = load_recognition_model()

# Function to recognize faces
def recognize_faces(image):
    # Convert to RGB (face_recognition uses RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return image, []
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # List to store recognition results
    results = []
    
    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Prepare face encoding for prediction
        face_encoding = np.array([face_encoding])
        
        # Make prediction
        prediction = model.predict(face_encoding)
        pred_idx = np.argmax(prediction[0])
        confidence = prediction[0][pred_idx]
        
        # Get name
        name = reverse_mapping[pred_idx]
        
        # Draw rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw label
        label = f"{name} ({confidence:.2f})"
        font_scale = image.shape[1] / 1000.0
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0]
        
        # Draw a filled rectangle for text background
        cv2.rectangle(image, (left, bottom - text_size[1] - 10), (left + text_size[0], bottom), (0, 255, 0), cv2.FILLED)
        
        # Put text
        cv2.putText(image, label, (left, bottom - 5), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 1)
        
        # Add to results
        results.append({
            "name": name,
            "confidence": float(confidence),
            "position": (top, right, bottom, left)
        })
    
    return image, results

# Function to record attendance
def record_attendance(recognized_names):
    # Create attendance directory if it doesn't exist
    if not os.path.exists("attendance"):
        os.makedirs("attendance")
    
    # Get current date for filename
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"attendance/{today}.csv"
    
    # Initialize DataFrame
    if os.path.exists(attendance_file):
        attendance_df = pd.read_csv(attendance_file)
    else:
        attendance_df = pd.DataFrame(columns=["Name", "Time", "Confidence"])
    
    # Current time
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Add new entries
    for name, confidence in recognized_names:
        # Check if person was already marked today
        if name in attendance_df["Name"].values:
            continue
            
        # Add new record
        new_record = pd.DataFrame({
            "Name": [name],
            "Time": [current_time],
            "Confidence": [confidence]
        })
        
        attendance_df = pd.concat([attendance_df, new_record], ignore_index=True)
    
    # Save updated attendance
    attendance_df.to_csv(attendance_file, index=False)
    
    return attendance_df

# Main content area
tab1, tab2 = st.tabs(["Face Recognition", "Attendance Records"])

with tab1:
    # Instructions
    st.header("Upload Image")
    st.markdown("Upload an image to recognize faces and mark attendance.")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Show original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        if model is not None:
            # Process button
            if st.button("Recognize Faces"):
                with st.spinner("Processing image..."):
                    # Start timing
                    start_time = time.time()
                    
                    # Recognize faces
                    result_image, results = recognize_faces(image.copy())
                    
                    # End timing
                    processing_time = time.time() - start_time
                    
                    # Show results
                    st.subheader("Recognition Result")
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Display processing time
                    st.info(f"Processing time: {processing_time:.2f} seconds")
                    
                    if results:
                        # Show detected individuals
                        st.subheader("Detected Individuals")
                        
                        # Create columns for each detected face
                        recognized_names = []
                        
                        for i, result in enumerate(results):
                            name = result["name"]
                            confidence = result["confidence"]
                            recognized_names.append((name, confidence))
                            
                            st.write(f"Person {i+1}: {name} (Confidence: {confidence:.2f})")
                        
                        # Record attendance
                        if st.button("Mark Attendance"):
                            attendance_df = record_attendance(recognized_names)
                            st.success("Attendance marked successfully!")
                            st.dataframe(attendance_df)
                    else:
                        st.warning("No faces detected in the image.")
        else:
            st.error("Please upload the model files first!")

with tab2:
    st.header("Attendance Records")
    
    # Check for attendance files
    if os.path.exists("attendance"):
        attendance_files = [f for f in os.listdir("attendance") if f.endswith(".csv")]
        
        if attendance_files:
            # Select date
            selected_date = st.selectbox("Select Date", attendance_files)
            
            # Display attendance for selected date
            if selected_date:
                attendance_path = os.path.join("attendance", selected_date)
                attendance_df = pd.read_csv(attendance_path)
                
                st.subheader(f"Attendance for {selected_date.split('.')[0]}")
                st.dataframe(attendance_df)
                
                # Download option
                st.download_button(
                    label="Download Attendance CSV",
                    data=attendance_df.to_csv(index=False),
                    file_name=selected_date,
                    mime="text/csv"
                )
        else:
            st.info("No attendance records found.")
    else:
        st.info("No attendance records found. Mark attendance first.")

# Footer
st.markdown("---")
st.markdown("Face Recognition Attendance System using CNN | Created for classroom demonstration")
