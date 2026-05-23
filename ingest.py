import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_knowledge_base():
    # 1. Use PDFMiner for better text reconstruction
    # Ensure this matches your filename in the /data folder!
    loader = PDFMinerLoader("./data/Makanaka CV.pdf.pdf")
    documents = loader.load()

    # 2. Split the text into cleaner chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # 3. Use the free local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create a FRESH Vector Database
    # We will save this in a folder called 'db_clean'
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./db_clean"
    )
    print(f"✅ CLEAN SUCCESS: Ingested {len(chunks)} chunks into 'db_clean'!")

if __name__ == "__main__":
    build_knowledge_base()