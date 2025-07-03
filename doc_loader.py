from langchain_community.document_loaders import PyMuPDFLoader,UnstructuredWordDocumentLoader,TextLoader
from RAG_Logic import pdf_rag,docx_rag,txt_rag



def pdf_loader(file_path):
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()
    pdf_vector_store=pdf_rag(docs) 
    return pdf_vector_store 
    
    
def docx_loader(file_path):
    loader = UnstructuredWordDocumentLoader(file_path=file_path)
    docs = loader.load()
    docx_vector_store = docx_rag(docs)
    return docx_vector_store
    
def txt_loader(file_path):
    loader = TextLoader(file_path=file_path)
    docs = loader.load()
    text_vector_store = txt_rag(docs)
    return text_vector_store

