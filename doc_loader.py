from langchain_community.document_loaders import PyMuPDFLoader,CSVLoader,UnstructuredWordDocumentLoader,json_loader,TextLoader
from RAG_Logic import pdf_rag,docx_rag,csv_rag,txt_rag,json_rag
from RAG_Logic import pdf_rag


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
    
    
def csv_docs_loader(file_path):
    loader = CSVLoader(file_path=file_path,encoding="utf-8-sig")
    docs = loader.load()
    csv_vector_store = csv_rag(docs)
    
    return csv_vector_store


def load_json_texts(file_path, jq_schema):
    loader = json_loader(file_path=file_path, jq_schema=jq_schema)
    docs = loader.load()
    json_vector_store = json_rag(docs)
    return json_vector_store

def txt_loader(file_path):
    loader = TextLoader(file_path=file_path)
    docs = loader.load()
    text_vector_store = txt_rag(docs)
    return text_vector_store

