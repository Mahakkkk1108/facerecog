import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Configure Streamlit page
st.set_page_config(
    page_title="Facial Expression Detection App",
    layout="centered"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #f63366;
    }
    .sub-header {
        font-size: 1.2em;
        color: #999999;
        text-align: center;
    }
    .label {
        font-weight: bold;
        color: #f63366;
    }
    .result-box {
        padding: 10px;
        background-color: #2d2d2d;
        color: #f2f2f2;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 1.1em;
    }
    .image-box {
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\hp\Downloads\DATA\emotion_recognition_model50.h5")
    return model

# Initialize model
try:
    model = load_model()
except Exception as e:
    st.error("Error loading model. Please ensure 'face_model.h5' is in the correct directory.")
    model = None

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'username' not in st.session_state:
    st.session_state.username = ''

def change_page(page):
    st.session_state.page = page

def preprocess_face_for_prediction(face_img):
    try:
        face_img = cv2.resize(face_img, (48, 48))
        if len(face_img.shape) == 2:  # If grayscale
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:  # If RGBA
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
        face_img = np.expand_dims(face_img, axis=0) / 255.0
        return face_img
    except Exception as e:
        st.error(f"Error preprocessing face: {str(e)}")
        return None

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def process_image(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:  # Grayscale image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image.copy()
        
        enhanced_image = enhance_image(image_rgb)
        image_for_display = image_rgb.copy()
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(500, 500)
        )
        
        results = []
        for (x, y, w, h) in faces:
            padding = int(0.1 * w)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image_rgb.shape[1], x + w + padding)
            y2 = min(image_rgb.shape[0], y + h + padding)
            face_roi = image_rgb[y1:y2, x1:x2]
            processed_face = preprocess_face_for_prediction(face_roi)
            
            if processed_face is not None and model is not None:
                prediction = model.predict(processed_face)
                emotion_idx = np.argmax(prediction[0])
                emotion_label = EMOTIONS[emotion_idx]
                confidence = float(prediction[0][emotion_idx])
                
                cv2.rectangle(image_for_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{emotion_label} ({confidence:.2%})"
                cv2.putText(image_for_display, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                results.append({
                    'emotion': emotion_label,
                    'confidence': confidence,
                    'position': (x1, y1, x2 - x1, y2 - y1)
                })
        
        return image_for_display, len(faces), results
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return image, 0, []

# Home Page
if st.session_state.page == 'home':
    st.markdown("<div class='main-title'>Facial Recognization 😉☺😉</div>", unsafe_allow_html=True)
    
    username = st.text_input("Enter your username:", key="username_input")
    
    if st.button("Continue"):
        if username:
            st.session_state.username = username
            change_page('detection')
        else:
            st.error("Please enter a username to continue")

# Detection Page
elif st.session_state.page == 'detection':
    st.title(f"Hey {st.session_state.username}!")
    st.write("Let's detect facial expressions in your images!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                processed_image, face_count, results = process_image(image)
                
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                if face_count > 0:
                    st.success(f"Detected {face_count} face{'s' if face_count != 1 else ''}")
                    for idx, result in enumerate(results, 1):
                        st.write(f"Face {idx}:")
                        st.write(f"- Emotion: {result['emotion']}")
                        st.write(f"- Confidence: {result['confidence']:.2%}")
                else:
                    st.warning("No faces detected in the image")
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
    
    with col2:
        st.subheader("Capture Image")
        picture = st.camera_input("Take a picture")
        
        if picture is not None:
            try:
                image = Image.open(picture)
                processed_image, face_count, results = process_image(image)
                
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                
                if face_count > 0:
                    st.success(f"Detected {face_count} face{'s' if face_count != 1 else ''}")
                    for idx, result in enumerate(results, 1):
                        st.markdown(f"<div class='result-box'><span class='label'>Face:</span> {idx}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='result-box'><span class='label'>Emotion:</span> {result['emotion']}</div>", unsafe_allow_html=True)
                        st.write(f"- Confidence: {result['confidence']:.2%}")
                else:
                    st.warning("No faces detected in the image")
            except Exception as e:
                st.error(f"Error processing captured image: {str(e)}")

    if st.button("Back to Home"):
        change_page('home')
