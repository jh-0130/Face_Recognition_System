import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dlib
import tempfile
import os
from PIL import Image
import io
import time

# Set page config
st.set_page_config(
    page_title="Celebrity Face Recognition",
    page_icon="🤖",
    layout="wide"
)

# Main application title
st.title("Celebrity Face Recognition")
st.write("Upload an image or use your webcam to identify celebrities")

# Load models function
@st.cache_resource
def load_models():
    """Load and cache the face detection and recognition models"""
    
    # Initialize Dlib's face detector
    detector = dlib.get_frontal_face_detector()
    
    # Create feature extractor
    feature_extractor = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    feature_extractor.trainable = False
    
    # Load your trained model here
    # Replace this with actual code to load your model
    # model = tf.keras.models.load_model('your_model_path.h5')
    
    # Placeholder for demo purposes
    if 'model' not in st.session_state:
        st.session_state.model_loaded = False
        st.warning("⚠️ This is a demo version. Please upload your trained model via the sidebar.")
    
    # For demo, we'll initialize reverse_mapping with sample celebrities
    # Replace this with your actual label mapping
    reverse_mapping = {
        0: "Akshay Kumar",
        1: "Brad Pitt",
        2: "Hugh Jackman",
        3: "Roger Federer",
        4: "Tom Cruise",
        # Add more celebrities from your actual model
    }
    
    return detector, feature_extractor, reverse_mapping

# Function to process a single image
def process_image(img, detector, feature_extractor, model, reverse_mapping, confidence_threshold=0.4):
    """Process an image and recognize faces"""
    # Convert to RGB (in case it's BGR from OpenCV)
    if len(img.shape) == 3 and img.shape[2] == 3:
        if isinstance(img, np.ndarray):
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = img
    else:
        st.error("Unsupported image format")
        return None, []
    
    # Create a copy for drawing
    result_img = rgb_img.copy()
    
    # Detect faces
    detected_faces = detector(rgb_img)
    
    if len(detected_faces) == 0:
        st.warning("No faces detected in the image.")
        return result_img, []
    
    results = []
    
    # Process each face
    for face_idx, face in enumerate(detected_faces):
        try:
            # Extract face coordinates
            top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
            
            # Add margin (15% on each side)
            height, width = bottom - top, right - left
            margin_h, margin_w = int(height * 0.15), int(width * 0.15)
            
            # Make sure margins don't go outside image bounds
            face_top = max(0, top - margin_h)
            face_bottom = min(rgb_img.shape[0], bottom + margin_h)
            face_left = max(0, left - margin_w)
            face_right = min(rgb_img.shape[1], right + margin_w)
            
            face_region = rgb_img[face_top:face_bottom, face_left:face_right]
            
            # Resize to target size (160x160)
            face_image = cv2.resize(face_region, (160, 160))
            
            # For the demo version without a model, generate random predictions
            if not st.session_state.get('model_loaded', False):
                # Just for demo - random predictions
                import random
                predictions = np.zeros(len(reverse_mapping))
                for i in range(len(predictions)):
                    predictions[i] = random.random()
                # Normalize
                predictions = predictions / np.sum(predictions)
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]
                predicted_name = reverse_mapping.get(predicted_class, "Unknown")
            else:
                # Actual prediction code (when model is loaded)
                # Convert to array and preprocess for MobileNetV2
                face_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                    np.expand_dims(np.array(face_image), axis=0)
                )
                
                # Get embedding
                face_embedding = feature_extractor.predict(face_array)
                face_embedding = face_embedding.reshape(1, -1)
                
                # Make prediction
                predictions = model.predict(face_embedding)[0]
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]
                predicted_name = reverse_mapping.get(predicted_class, "Unknown")
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                predicted_name = "Unknown"
            
            # Get top 3 predictions for display
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [(reverse_mapping.get(idx, f"Class {idx}"), 
                               float(predictions[idx])) for idx in top_indices]
            
            # Store the results
            results.append({
                "face_idx": face_idx,
                "bbox": (left, top, right, bottom),
                "predicted_name": predicted_name,
                "confidence": float(confidence),
                "top_predictions": top_predictions,
                "face_image": face_image
            })
            
            # Draw rectangle and name
            color = (0, 255, 0) if confidence >= confidence_threshold else (255, 0, 0)
            cv2.rectangle(result_img, (left, top), (right, bottom), color, 2)
            
            label_text = f"{predicted_name} ({confidence:.2f})"
            font_scale = result_img.shape[1] / 1000.0
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)[0]
            
            # Place text above the face if possible
            if top > text_size[1] + 10:
                label_y = top - 10
            else:
                label_y = bottom + text_size[1] + 10
            
            cv2.putText(result_img, label_text, (left, label_y),
                      cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
            
        except Exception as e:
            st.error(f"Error processing face {face_idx+1}: {str(e)}")
    
    return result_img, results

# Main app function
def main():
    # Load models
    detector, feature_extractor, reverse_mapping = load_models()
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model uploader
    st.sidebar.subheader("Upload your Model")
    model_file = st.sidebar.file_uploader("Upload your trained face recognition model (.h5)", type=["h5"])
    if model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            tmp.write(model_file.getvalue())
            model_path = tmp.name
        
        try:
            model = tf.keras.models.load_model(model_path)
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            st.session_state.model_loaded = False
    
    # Confidence threshold adjustment
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.4, 
        step=0.05,
        help="Minimum confidence required for a face to be recognized"
    )
    
    # Get model from session state if available
    model = st.session_state.get('model', None)
    
    # Main content area
    tab1, tab2 = st.tabs(["📷 Webcam", "🖼️ Upload Image"])
    
    # Webcam tab
    with tab1:
        st.header("Webcam Face Recognition")
        
        # Start webcam capture
        webcam_container = st.container()
        with webcam_container:
            webcam_placeholder = st.empty()
            webcam_stopped = st.button("Stop Webcam", key="stop_webcam")
            
            run_webcam = st.checkbox("Start Webcam", value=False)
            
            if run_webcam and not webcam_stopped:
                # Use Streamlit's webcam input
                webcam_img = st.camera_input("Take a picture")
                
                if webcam_img is not None:
                    # Convert to OpenCV format
                    img = np.array(Image.open(webcam_img))
                    
                    # Process the image
                    result_img, face_results = process_image(
                        img, detector, feature_extractor, model, 
                        reverse_mapping, confidence_threshold
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(result_img, caption="Recognition Results", use_column_width=True)
                    
                    with col2:
                        if not face_results:
                            st.warning("No faces detected or recognized")
                        else:
                            for result in face_results:
                                with st.expander(f"Face #{result['face_idx']+1}: {result['predicted_name']}"):
                                    st.image(result['face_image'], caption="Extracted Face", width=160)
                                    
                                    # Display top predictions
                                    for name, conf in result['top_predictions']:
                                        # Create a progress bar for confidence
                                        st.write(f"{name}: {conf:.4f}")
                                        st.progress(float(conf))
    
    # Upload image tab
    with tab2:
        st.header("Upload an Image")
        
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert to OpenCV format
            img = np.array(Image.open(uploaded_file))
            
            # Process the image
            result_img, face_results = process_image(
                img, detector, feature_extractor, model, 
                reverse_mapping, confidence_threshold
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(result_img, caption="Recognition Results", use_column_width=True)
            
            with col2:
                if not face_results:
                    st.warning("No faces detected or recognized")
                else:
                    for result in face_results:
                        with st.expander(f"Face #{result['face_idx']+1}: {result['predicted_name']}"):
                            st.image(result['face_image'], caption="Extracted Face", width=160)
                            
                            # Display top predictions
                            for name, conf in result['top_predictions']:
                                # Create a progress bar for confidence
                                st.write(f"{name}: {conf:.4f}")
                                st.progress(float(conf))

    # Information and instructions
    st.sidebar.markdown("---")
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown("""
    1. Upload your trained model (.h5 file) in the sidebar
    2. Use the webcam tab to capture images or upload them
    3. Adjust the confidence threshold as needed
    4. The app will recognize celebrities based on your model
    """)
    
    # About
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this app**
    
    This app uses a deep learning model to recognize celebrities in images. 
    It uses MobileNetV2 for feature extraction and your custom trained model for classification.
    """)

if __name__ == "__main__":
    main()
