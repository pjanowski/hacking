# LoadVectorize.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import os


def load_doc(fileurl: str = None) -> list:
    # 600pgs, 3403 chunks, 40 seconds
    # loader = OnlinePDFLoader("https://support.riverbed.com/bin/support/download?did=b42r9nj98obctctoq05bl2qlga&version=9.14.2a")
    # 33pgs, 350 chunks, 10 seconds
    # loader = OnlinePDFLoader("https://arxiv.org/pdf/2401.08406.pdf")
    loader = OnlinePDFLoader(fileurl)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs


def vectorize(docs, index_path, embeddings=None, force_rebuild=False):
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings()

    if force_rebuild:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(index_path)
    else:
        db = FAISS.load_local(index_path, embeddings)

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    return db, bm25_retriever


def load_db(fileurl: str, index_path: str):
    if fileurl == "https://arxiv.org/pdf/2401.08406.pdfa":
        import pickle

        with open("docs.pkl", "rb") as filehandler:
            docs = pickle.load(filehandler)
    else:
        docs = load_doc(fileurl)
    if os.path.exists(index_path):
        print("Index exists, loading from disk...")
        db, bm25_retriever = vectorize(docs, index_path, force_rebuild=False)
    else:
        print("Exception: no index on disk, creating new...")
        db, bm25_retriever = vectorize(docs, index_path, force_rebuild=True)

    return db, bm25_retriever
