# Q&A Chatbot
#from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap

import google.generativeai as genai

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text


st.set_page_config("BASIC question Answering")

st.header("Gemini LLM APPLICATION")

input=st.text_input("input",key=input)
submit=st.button("Ask me a Question")


if submit:
    response=get_gemini_response(input)
    st.write(response)