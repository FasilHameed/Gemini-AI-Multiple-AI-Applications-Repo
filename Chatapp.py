# Q&A Chatbot
from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure GenAI with Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize GenerativeModel
model = genai.GenerativeModel("gemini-pro")

# Function to get response from Gemini
def get_gemini_response(question):
    response = model.start_chat(history=[]).send_message(question, stream=True)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="AI Chatbot")

# Page header and title with decoration
st.title("AI Smart Chatbot :speech_balloon:")

# Add some style to the UI
st.markdown("---")
st.info("Welcome to the AI Chatbot! Feel free to ask any questions.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Input text box with placeholder and key
input_question = st.text_input("Ask a question:", key="input", placeholder="Type your question here...")

# Button to trigger question with custom style
submit_button = st.button("Ask", key="submit", help="Click to ask your question")

# Display response with animated loading spinner
if submit_button and input_question:
    with st.spinner("Thinking..."):
        response = get_gemini_response(input_question)
        st.subheader("Response:")
        for chunk in response:
            if hasattr(chunk, "role"):
                if chunk.role == "system":
                    st.write(chunk.text)
                else:
                    st.write(f"**Bot:** {chunk.text}")
            else:
                st.write(f"**Bot:** {chunk.text}")
        # Update chat history
        st.session_state["chat_history"].append(("You", input_question))
        for chunk in response:
            if hasattr(chunk, "role"):
                if chunk.role == "system":
                    st.session_state["chat_history"].append(("Bot", chunk.text))
            else:
                st.session_state["chat_history"].append(("Bot", chunk.text))

# Decorate chat history section
st.subheader("Chat History :scroll:")
for role, text in st.session_state["chat_history"]:
    st.write(f"{role.capitalize()}: {text}")

# 
st.markdown("---")
st.warning("Remember, this is a demo version and may not provide accurate responses.")

# Footer
footer_with_image_light_blue = """
<style>
.footer {
    background-color: #E0F2F1;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

.footer img {
    max-width: 100%;
    border-radius: 10px;
    margin-top: 10px;
}

.footer .line {
    height: 1px;
    background-color: #B2DFDB;
    margin: 10px 0;
}

.footer .connect-text {
    color: #004D40;
    font-weight: bold;
    margin-bottom: 10px;
}

.footer a {
    margin: 0 10px;
}

.footer .powered-by {
    color: #004D40;
    font-size: 14px;
    margin-top: 10px;
}

.bright-text {
    color: #004D40;
}

/* Add Animation */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.chat-message {
    animation: fadeIn 0.5s ease-out;
}
</style>
<div class="footer">
    <div class="line"></div>
    <div class="connect-text">Connect with me at</div>
    <a href="https://github.com/FasilHameed" target="_blank"><img src="https://img.icons8.com/plasticine/30/000000/github.png" alt="GitHub"></a>
    <a href="https://www.linkedin.com/in/faisal--hameed/" target="_blank"><img src="https://img.icons8.com/plasticine/30/000000/linkedin.png" alt="LinkedIn"></a>
    <a href="tel:+917006862681"><img src="https://img.icons8.com/plasticine/30/000000/phone.png" alt="Phone"></a>
    <a href="mailto:faisalhameed763@gmail.com"><img src="https://img.icons8.com/plasticine/30/000000/gmail.png" alt="Gmail"></a>
    <div class="line"></div>
    <div class="powered-by">Powered By <img src="https://img.icons8.com/clouds/30/000000/gemini.png" alt="Gemini"> Gemini 💫 and Streamlit 🚀</div>
</div>
"""

# Render Footer
st.markdown(footer_with_image_light_blue, unsafe_allow_html=True)
