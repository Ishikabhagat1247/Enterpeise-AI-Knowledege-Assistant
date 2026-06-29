import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables (still kept for your app's GPT text generation)
load_dotenv()

DOCS_DIR = "documents"
CHROMA_DIR = "chroma_db"

def run_ingestion():
    print("Checking for documents...")
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print(f"⚠️ The '{DOCS_DIR}' directory is empty or missing. Please add some PDF files first!")
        return

    # 1. Document Loading
    print("Loading PDF documents...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw pages.")

    # 2. Chunking Strategy
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Local Embedding Generation & Storage
    print("Initializing Local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
    # This downloads and runs completely free on your local machine
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Indexing text chunks into ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print(f"🎉 Successfully ingested and indexed chunks to local database '{CHROMA_DIR}'!")

if __name__ == "__main__":
    run_ingestion()