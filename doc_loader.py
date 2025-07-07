from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader,TextLoader
from RAG_Logic import create_vector



def pdf_loader(file_path):
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()
    pdf_vector_store = create_vector(docs) 
    return pdf_vector_store 
    
    
def docx_loader(file_path):
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    return create_vector(docs)
    
def txt_loader(file_path):
    loader = TextLoader(file_path=file_path)
    docs = loader.load()
    text_vector_store = create_vector(docs)
    return text_vector_store

