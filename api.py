import os
import shutil
import csv
from io import StringIO
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from dotenv import load_dotenv

# Core LangChain, Driver & Integration Libraries
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import pdfplumber  

# Load environment configuration variables securely
load_dotenv()

app = FastAPI(title="IntellectRAG Verification Engine API")

# Initialize Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Path for temporary handling of incoming files
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =====================================================================
# 1. ENVIRONMENT VALIDATION & CONFIGURATION
# =====================================================================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("CRITICAL ERROR: MONGO_URI missing from environment setup (.env)")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("CRITICAL ERROR: GROQ_API_KEY missing from environment setup (.env)")

# =====================================================================
# 2. HIGH-PERFORMANCE API EMBEDDING ROUTE (LOW MEMORY PROFILE)
# =====================================================================
client = MongoClient(MONGO_URI)
MONGODB_COLLECTION = client["resume_rag"]["embeddings"]
ATLAS_VECTOR_INDEX_NAME = "vector_index"

# Routes embedding execution through Groq's API pipeline to keep memory minimal
embeddings = OpenAIEmbeddings(
    model="nomic-embed-text-v1.5",
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1"
)

# Active MongoDB vector store bridge
vector_db = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_INDEX_NAME,
    embedding_key="embedding",
    text_key="text"
)

# =====================================================================
# 3. CLOUD TEXT GENERATION PIPELINE SETUP (GROQ LLM)
# =====================================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",  
    temperature=0.0,
    groq_api_key=GROQ_API_KEY
)

# =====================================================================
# 4. CONVERSATIONAL MEMORY STORAGE & PROMPT TEMPLATE
# =====================================================================
chat_history = []

template = """<|im_start|>system
You are an expert HR recruitment compliance auditor. Your sole task is to extract information truthfully and accurately from the provided Context based on the User's question.

CRITICAL INSTRUCTIONS:
1. Examine the context closely for contact information, locations, qualifications, and specific degree terms (e.g., BSc, Information Technology, Diploma, etc.).
2. If the user asks for an address, location, or where someone stays, look for city/country combinations (like "Durban, South Africa") or street names, and provide exactly that. Do NOT substitute a phone number or email address for a physical location.
3. If the context explicitly states a qualification or field of study, extract it clearly.
4. If the requested information is genuinely missing from the provided context blocks, answer exactly: "Not explicitly stated."

Context:
{context}

History:
{history}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
Answer:"""
prompt = PromptTemplate.from_template(template)


# =====================================================================
# 5. API ROUTE PATHS
# =====================================================================

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Serves the interactive frontend dashboard cleanly."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    LAYOUT-AWARE BATCH UPLOADER: Extracts text line-by-line, strips whitespace,
    and aggressively filters out any null or blank text segments to prevent
    empty inputs from crashing the remote embedding API.
    """
    # Clear the collection before saving the newly uploaded batch profiles
    MONGODB_COLLECTION.delete_many({})
    
    total_chunks_processed = 0
    processed_filenames = []
    
    for file in files:
        # Clean up double extensions if they accidentally occur (e.g., Makanaka CV.pdf.pdf)
        clean_filename = file.filename
        if clean_filename.count('.pdf') > 1:
            clean_filename = clean_filename.replace('.pdf.pdf', '.pdf')
            
        if not clean_filename.lower().endswith(('.pdf', '.txt')):
            continue  
            
        file_path = os.path.join(UPLOAD_DIR, clean_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        extracted_text = ""
        try:
            if clean_filename.lower().endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        try:
                            words = page.extract_words(initialize_with_options=True) or page.extract_words()
                        except TypeError:
                            words = page.extract_words()
                        
                        if words:
                            lines = {}
                            for w in words:
                                top = round(w['top'], 1)
                                matched_top = min(lines.keys(), key=lambda t: abs(t - top), default=None)
                                
                                if matched_top is not None and abs(matched_top - top) < 3:
                                    lines[matched_top].append(w)
                                else:
                                    lines[top] = [w]
                            
                            page_lines = []
                            for t in sorted(lines.keys()):
                                sorted_words = sorted(lines[t], key=lambda w: w['x0'])
                                line_text = " ".join([w['text'] for w in sorted_words])
                                page_lines.append(line_text)
                                
                            extracted_text += "\n".join(page_lines) + "\n"
                        else:
                            text = page.extract_text()
                            if text:
                                extracted_text += text + "\n"
                                
            elif clean_filename.lower().endswith('.txt'):
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                    
            # Skip file entirely if no text content was found
            if not extracted_text or not extracted_text.strip():
                continue
            
            processed_lines = []
            for line in extracted_text.split("\n"):
                if line.strip():
                    processed_lines.append(line.strip())
            extracted_text = "\n".join(processed_lines)
            
            if not extracted_text.strip():
                continue
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,       
                chunk_overlap=150,    
                separators=["\n\n", "\n", " ", ""]
            )
            
            split_texts = text_splitter.split_text(extracted_text)
            
            # CRITICAL ADVANCED VALIDATION: Ensure text is non-empty and stripped 
            # to guarantee the remote API receives a clean, non-blank string element.
            chunks = []
            for text in split_texts:
                cleaned_chunk_text = text.strip()
                if cleaned_chunk_text: # Strictly filters out empty strings or structural gaps
                    chunks.append(Document(page_content=cleaned_chunk_text, metadata={"source": clean_filename.lower()}))
            
            if chunks:
                vector_db.add_documents(chunks)
                total_chunks_processed += len(chunks)
                processed_filenames.append(clean_filename)
                
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error parsing {clean_filename}: {str(e)}")
        finally:
            if os.path.exists(file_path): 
                os.remove(file_path)
                
    if not processed_filenames:
        raise HTTPException(status_code=400, detail="No valid text content could be processed from the uploaded documents.")
        
    return {
        "status": "Success",
        "files_processed": processed_filenames,
        "chunks_processed": total_chunks_processed,
        "message": "Batch vector processing completed successfully with blank-string safety controls!"
    }


@app.get("/ask")
def ask_ai(question: str):
    """
    HYBRID SEARCH FILTERED QUERY: Automatically extracts target candidates,
    runs vector matches, and seamlessly layers regex structural lookups to ensure
    precise contact information and qualification retrieval.
    """
    global chat_history
    lowered_q = question.lower()
    
    target_candidate = None
    if "makanaka" in lowered_q:
        target_candidate = "makanaka"
    elif "orripah" in lowered_q:
        target_candidate = "orripah"
        
    search_filter = {}
    if target_candidate:
        search_filter = {"metadata.source": {"$regex": target_candidate, "$options": "i"}}
        
    docs = []
    try:
        if search_filter:
            docs = vector_db.similarity_search(question, k=5, pre_filter=search_filter)
        else:
            docs = vector_db.similarity_search(question, k=5)
    except Exception:
        docs = vector_db.similarity_search(question, k=4)
        
    fallback_keywords = []
    if any(kw in lowered_q for kw in ["address", "location", "stay", "live", "where"]):
        fallback_keywords.extend(["durban", "gauteng", "south africa", "road", "street", "avenue", "residential"])
    if any(kw in lowered_q for kw in ["phone", "cell", "number", "contact", "call"]):
        fallback_keywords.extend(["062", "072", "082", "061", "063", "064", "065", "+27", "tel"])
    if any(kw in lowered_q for kw in ["degree", "diploma", "qualification", "university", "college", "study", "studied", "attend"]):
        fallback_keywords.extend(["bsc", "bachelor", "diploma", "university", "nwu", "north-west", "technology", "information"])

    if fallback_keywords:
        regex_pattern = "|".join(fallback_keywords)
        fallback_query = {"text": {"$regex": regex_pattern, "$options": "i"}}
        
        if target_candidate:
            fallback_query["metadata.source"] = {"$regex": target_candidate, "$options": "i"}
            
        try:
            fallback_cursor = MONGODB_COLLECTION.find(fallback_query).limit(3)
            for item in fallback_cursor:
                doc_text = item.get("text", "")
                if doc_text and not any(doc_text.strip() == d.page_content.strip() for d in docs):
                    docs.append(Document(page_content=doc_text))
        except Exception:
            pass 
            
    context_chunks = [doc.page_content.strip() for doc in docs if doc.page_content]
    context = "\n---\n".join(context_chunks)
    
    if not context.strip():
        return {
            "question": question,
            "answer": "Not explicitly stated.",
            "sources": []
        }
        
    history_str = ""
    for turn in chat_history[-2:]:
        history_str += f"User: {turn['q']}\nAI: {turn['a']}\n"
    if not history_str:
        history_str = "None"
        
    formatted_prompt = prompt.format(context=context, history=history_str, question=question)
    
    ai_response = llm.invoke(formatted_prompt)
    final_answer = ai_response.content
    
    if "Answer:" in final_answer:
        final_answer = final_answer.split("Answer:")[-1]
    final_answer = final_answer.split("<|im_end|>")[0].split("User:")[0].strip()
    
    chat_history.append({"q": question, "a": final_answer})
    
    return {
        "question": question,
        "answer": final_answer,
        "sources": context_chunks  
    }


@app.get("/export")
def export_history_to_csv():
    """Streams evaluation memory logs directly into a downloadable CSV file format."""
    global chat_history
    if not chat_history:
        raise HTTPException(status_code=400, detail="No evaluation history available to export.")
        
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Evaluation Question", "Verified AI Compliance Output"])
    
    for turn in chat_history:
        writer.writerow([turn["q"], turn["a"]])
        
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=compliance_report.csv"
    return response


@app.post("/clear_chat")
def clear_chat():
    """Clears conversational memory structures completely."""
    global chat_history
    chat_history = []
    return {"status": "Memory reset complete"}


    