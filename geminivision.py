from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# Configure GenAI with Google API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize GenerativeModel
model = genai.GenerativeModel('gemini-pro-vision')

# Function to get response from Gemini
def get_gemini_response(input_text, image):
    if input_text != "":
        response = model.generate_content([input_text, image])
        return response.text
    else:
        return ""

# Set Streamlit page title
st.set_page_config(page_title="Gemini Vision Application")

# Page header
st.header("Vision AI")

# Input prompt for text
input_text = st.text_input("Input Prompt:", key="input")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Button to trigger inference
submit_button = st.button("Tell me about the image")


# Trigger response generation on button click
if submit_button:
    response = get_gemini_response(input_text, image)
    st.write(response)
