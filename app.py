from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the local database we just created
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./db_clean", embedding_function=embeddings)

def ask_cv(question):
    # Search the database for the most relevant parts of your CV
    docs = vector_db.similarity_search(question, k=2)
    
    print(f"\n--- Information found in your CV ---")
    for doc in docs:
        print(f"-> {doc.page_content}\n")

if __name__ == "__main__":
    query = input("Ask a question about your career: ")
    ask_cv(query)