from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# from api_key import openai_api_key
from dotenv import load_dotenv
import os
# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_retriever():
    embed = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    txts = ['intro1.txt','intro2.txt','intro3.txt','intro4.txt']
    docs = [TextLoader('docs/'+txt).load() for txt in txts]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorDB = FAISS.from_documents(
        documents=doc_splits,
        embedding=embed
    )
    retriever = vectorDB.as_retriever()
    return retriever