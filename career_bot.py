import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Setup the Free LLM (Replace with your Groq key)
os.environ["GROQ_API_KEY"] = "gsk_kdduUQDvDUxFShoWWbHougyHjr0tFz3E38fX8e0bnTUpya-P0mXW"
llm = ChatGroq(model_name="llama3-8b-8192")

# 2. Load your clean database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./db_clean", embedding_function=embeddings)

# 3. Create the "Chain" that links the DB to the AI
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

# 4. Ask a natural question
query = "Based on the CV, why should I hire Makanaka for a Backend role?"
response = rag_chain.invoke(query)

print("\n--- AI Career Assistant Response ---")
print(response["result"])