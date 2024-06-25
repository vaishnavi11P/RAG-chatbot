import os
import shutil
import time
import streamlit as st
from PyPDF2 import PdfReader
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.llms import Replicate
from dotenv import load_dotenv

load_dotenv()

# CONSTANTS
DB_PATH = "db/"
MAX_RETIRES = 3
RETRY_DELAY = 1
server_thread = None


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def get_ollama_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-pro")


def get_ollama_llm():
    return ChatOllama(model="llama3")


def get_replicate_llm():
    return Replicate(model="meta/meta-llama-3-70b-instruct")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, path: str):
    embeddings = get_embeddings()
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=path)
    return vector_store


def get_conversational_chain(retriever):
    llm = get_llm()

    prompt_template = """You are a helpful assistant providing detailed, accurate, and informative responses. Your 
    task is to assist users by providing relevant information retrieved from a set of documents and generating 
    coherent and contextually appropriate responses based on the retrieved information.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:"""
    prompt_template = ChatPromptTemplate.from_template(prompt_template)

    # Create chain
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )

    return chain


def user_input(user_question):
    embeddings = get_embeddings()
    new_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = new_db.as_retriever(search_kwargs={"k": 2})

    chain = get_conversational_chain(retriever)

    response_container = st.empty()  # Create an empty container for real-time updates

    response = ""
    for chunk in chain.stream(user_question):
        response += chunk
        response_container.markdown(response)  # Update the markdown in real-time


def delete_and_recreate_db_directory():
    try:
        shutil.rmtree(DB_PATH)
        print(f"Contents of '{DB_PATH}' successfully cleared.")
    except OSError as e:
        print(f"Error: {DB_PATH} : {e.strerror}")
        return False

    try:
        os.makedirs(DB_PATH)
        print(f"Empty directory '{DB_PATH}' successfully created.")
        return True
    except OSError as e:
        print(f"Error: {DB_PATH} : {e.strerror}")
        return False


def retries():
    # ! Logic to delete the db and create a new one with fresh embeddings
    for attempt in range(MAX_RETIRES):
        if delete_and_recreate_db_directory():
            break

        print(f"Retry {attempt + 1} in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
    else:
        print("Max retries exceeded. Could not delete or recreate directory.")


class QueryRequest(BaseModel):
    query: str


def main():
    st.set_page_config("Chat Docx")
    st.header("Just a normal RAG chatbot")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        submit_button = st.button("Submit & Process")

        if submit_button:
            if len(pdf_docs) == 0:
                st.warning("Upload a document", icon='⚠️')
            else:
                with st.spinner("Processing..."):
                    retries()
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, path=DB_PATH)
                    st.success("Vector store created and data processed.")


if __name__ == "__main__":
    main()