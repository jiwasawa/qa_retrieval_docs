import datetime
import openai
import os
import streamlit as st
import tempfile
import uuid

from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Qdrant
from openai.error import InvalidRequestError

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


def load_pdf(pdf):
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    return docs


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


def create_vectordb_for_docs(docs, openai_api_key, dir_name, db="qdrant"):
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
            collection_name=dir_name,
        )
    elif db == "chroma":
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=f"docs/chroma/{dir_name}",
        )
    else:
        return
    return vectordb, embedding


def init_llm(openai_api_key):
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=openai_api_key)
    return llm


def init_qa_retriever(docs, openai_api_key, use_compression=True, use_memory=False):
    llm = init_llm(openai_api_key)  # init gpt-3.5-turbo
    dir_name = os.path.splitext(
        os.path.basename(docs[0].metadata["source"])
    )[0].replace(" ", "_")  # use filename as directory name
    vectordb, embedding = create_vectordb_for_docs(docs, openai_api_key, dir_name)
    if use_compression:
        compressor = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.5)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectordb.as_retriever(search_type="mmr"),
        )
    else:
        retriever=vectordb.as_retriever(search_type="mmr")
    if use_memory:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory,
        )
    else:
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )
        qa = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )
    return qa, retriever


st.set_page_config(page_title="QA_youtube_pdf")
st.title("QA with YouTube or PDF")

url = st.text_input(
    "Enter YouTube url (preprocessing might take a few minutes):",
    placeholder="https://www.youtube.com/watch?v=nM_3d37lmcM",
)
pdf = st.file_uploader("Upload an article:", type="pdf")
spinner_msg = (
    "Downloading & transcribing the video... This might take a few minutes." 
    if url else "Processing pdf"
)
keys = {}
USE_MEMORY = False
if USE_MEMORY:
    keys["question"] = "question"
    keys["answer"] = "answer"
else:
    keys["question"] = "query"
    keys["answer"] = "result"
if "api_key" not in st.session_state:
    # require OPENAI_API_KEY
    try:
        st.session_state.api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        st.session_state.api_key = st.text_input('OpenAI API Key', type='password')

with st.form("upload_form", clear_on_submit=False):
    disable_cond = ("docs" in st.session_state) or not((url or pdf) and st.session_state.api_key)
    submitted = st.form_submit_button("Preprocess YouTube/PDF", disabled=disable_cond)
    summary_storage = []
    if submitted:
        openai.api_key = st.session_state.api_key
        with st.spinner(spinner_msg):
            if pdf:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(pdf.getvalue())
                    tmp_file_path = tmp_file.name
                    st.session_state.docs = load_pdf(tmp_file_path)
            elif url:
                st.session_state.docs = load_youtube(url)

if "summary" in st.session_state:
    st.info("Summary: " + st.session_state.summary)
else:
    with st.form("summary_form", clear_on_submit=False):
        summary_required = st.form_submit_button(
            "Summarize document",
            help="Generates summary (this can be skipped)",
            disabled=not((url or pdf) and st.session_state.api_key and ("docs" in st.session_state)),
        )
        if summary_required:
            with st.spinner("Summarizing... (might take a few minutes)"):
                st.session_state.summary = summarize_doc(st.session_state.docs, st.session_state.api_key)
            st.info(st.session_state.summary)

if "responses" not in st.session_state:
    st.session_state.responses = []
    st.session_state.sources = []
else:
    for response in st.session_state.responses:
        st.info("Q. " + response[keys["question"] ])
        st.info("A. " + response[keys["answer"] ])

question = st.text_input(
    "Enter your question:",
    placeholder="Is there any information about X?",
)
with st.form("question_form", clear_on_submit=True):
    q_submitted = st.form_submit_button("Ask question", disabled=not (question and ("docs" in st.session_state)))
    if q_submitted:
        with st.spinner("Asking question"):
            qa, retriever = init_qa_retriever(
                st.session_state.docs, 
                st.session_state.api_key,
            )
            response = qa({keys["question"]: question})
            st.session_state.responses.append(response)
            st.session_state.sources.append(retriever.get_relevant_documents(query=question))
        
        st.info("A. " + st.session_state.responses[-1][keys["answer"]])
        st.info("Sources for answering 1/4: \n" + st.session_state.sources[-1][0].page_content)
        st.info("Sources for answering 2/4: \n" + st.session_state.sources[-1][1].page_content)

with st.form("new_question", clear_on_submit=True):
    q_submitted = st.form_submit_button("Ask another question", disabled=not q_submitted)
