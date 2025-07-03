#RAG ka logic.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter



def pdf_rag(docs):    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pdf_vector_store = FAISS.from_documents(chunks, embedding_model)
    return pdf_vector_store
    
def csv_rag(docs):    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    csv_vector_store = FAISS.from_documents(chunks, embedding_model)
    return csv_vector_store

def json_rag(docs):    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    json_vector_store = FAISS.from_documents(chunks, embedding_model)
    return json_vector_store

def txt_rag(docs):    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    txt_vector_store = FAISS.from_documents(chunks, embedding_model)
    return txt_vector_store

def docx_rag(docs):    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docx_vector_store = FAISS.from_documents(chunks, embedding_model)
    return docx_vector_store

    
    