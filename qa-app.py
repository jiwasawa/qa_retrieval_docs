import datetime
import openai
import os
import streamlit as st

from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Qdrant

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


def load_youtube(url: str, save_dir="docs/youtube/"):
    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    return docs


def summarize_doc(docs, openai_api_key):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)
    try:
        chain = load_summarize_chain(llm, chain_type="stuff")
        summary = chain.run(docs)
    except InvalidRequestError:
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
    return summary


def create_vectordb_for_docs(docs, openai_api_key, db="qdrant"):
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )
    splits = text_splitter.split_documents(docs)
    if db == "qdrant":
        vectordb = Qdrant.from_documents(
            documents=splits, 
            embedding=embedding,
            path="./docs/qdrant",
            collection_name="youtube_docs"
        )
    elif db == "chroma":
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory="docs/chroma/",
        )
    else:
        return
    return vectordb


def init_llm(openai_api_key):
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=openai_api_key)
    return llm


st.set_page_config(page_title="QA youtube transcription")
st.title("QA youtube transcription")


url = st.text_input("Enter YouTube url:", placeholder="https://www.youtube.com/watch?v=nM_3d37lmcM")
question = st.text_input("Enter your question:", placeholder="What does Schulman think is important for the future of ChatGPT?", disabled=not url)
openai_api_key = st.text_input('OpenAI API Key', type='password')

result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(url and question and openai_api_key))
    if submitted:
        openai.api_key = openai_api_key
        with st.spinner("Downloading & transcribing the video... This might take a few minutes."):
            docs = load_youtube(url)
            #llm = init_llm(openai_api_key)  # init gpt-3.5-turbo
            #vectordb = create_vectordb_for_docs(docs, openai_api_key)
            #retriever=vectordb.as_retriever()
            #memory = ConversationBufferMemory(
            #    memory_key="chat_history",
            #    return_messages=True
            #)
            #qa = ConversationalRetrievalChain.from_llm(
            #    llm,
            #    retriever=retriever,
            #    memory=memory,
            #)
            #response = qa({"question": question})
            #result.append(response)
            summary = summarize_doc(docs, openai_api_key)
if len(result):
    #st.info(response["answer"])
    st.info(summary)
