import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


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


if __name__ == "__main__":
    main()
