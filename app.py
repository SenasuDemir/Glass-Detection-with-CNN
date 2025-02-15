import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('cnn_model.h5', compile=False)

# Function to process the uploaded image
def process_image(img):
    img = img.resize((64, 64))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Custom CSS for better design
st.markdown("""
    <style>
        .main { 
            background-color: #f0f4f7; 
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #4caf50;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
        }
        .description {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #4caf50;
            color: white;
            font-size: 18px;
            border-radius: 5px;
            padding: 10px 20px;
            margin-top: 20px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

# Title of the application
st.title('Glass Detection from Image 📸')

# Brief description
st.markdown('<p class="description">Upload a photo, and the model will predict whether there is a glass or not.</p>', unsafe_allow_html=True)

# File uploader for the user to upload an image
file = st.file_uploader('Select an image (jpg, jpeg, png)', type=['jpg', 'jpeg', 'png'])

if file is not None:
    # Displaying the uploaded image
    img = Image.open(file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Processing the image
    image = process_image(img)
    prediction = model.predict(image)

    # Get the predicted class index
    predicted_class = np.argmax(prediction, axis=-1)

    # Displaying prediction result
    st.subheader("Prediction Result:")

    if predicted_class == 1:
        prediction_text = '✅ There is a glass'
    else:
        prediction_text = '❌ There is no a glass'

    st.write(prediction_text)

else:
    st.write("Please upload an image to get started.")

# Footer
st.markdown("<p style='text-align: center; font-size: 12px; color: #888;'>Built with 💚 by Senasu Demir</p>", unsafe_allow_html=True)
