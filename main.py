# from dotenv import load_dotenv
# load_dotenv()
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import tempfile
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
# from PyPDF2 import PdfMerger
loader = PyPDFLoader("output.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
texts = text_splitter.split_documents(pages)
embeddings_model = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings_model)

st.title("리니지2M 23년 9월 업데이트 검색")
st.write("---")

question = st.text_input('질문해주세요', '입력 창')

if st.button('검색'):
    with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write('질문 내용: ', question)
            st.write(result["result"])
    st.success('Done!')



def pdf_to_document(uploaded_file):
    # Read documents
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages
    
# uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
# if uploaded_file is not None:
    # pages = pdf_to_document(uploaded_file)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 800,
    #     chunk_overlap  = 20,
    #     length_function = len,
    #     is_separator_regex = False,
    # )
    # texts = text_splitter.split_documents(pages)

    # embeddings_model = OpenAIEmbeddings()
    # # load it into Chroma
    # db = Chroma.from_documents(texts, embeddings_model)

    # question = st.text_input('pdf 관련 질문해주세요', '입력 창')

    # if st.button('검색'):
    #     with st.spinner('Wait for it...'):
    #             llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    #             qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
    #             result = qa_chain({"query": question})
    #             st.write('질문 내용: ', question)
    #             st.write(result["result"])

    #     st.success('Done!')



# merger = PdfMerger()
# pdf_dir = os.getcwd()
# for filename in os.listdir(pdf_dir):
#     if filename.endswith('.pdf'):
#          merger.append(filename)
# merger.write("output.pdf")
# merger.close()

