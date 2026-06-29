import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Enterprise Knowledge Engine - REST API Backend",
    description="Production-grade REST endpoints for multi-document RAG queries under verification guardrails.",
    version="1.0.0"
)

DB_DIR = "chroma_db"
EXPECTED_TOKEN = "admin123"

# Setup security header check (X-API-Key: admin123)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# 2. Data Transfer Objects (Schemas)
class QueryInput(BaseModel):
    question: str = Field(..., example="What are the main architectural constraints of the system?")

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list = Field(default_factory=list)

# 3. Core Resource Initialization
# Load the exact same embedding model used during ingestion to maintain vector consistency
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(DB_DIR):
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
else:
    db = None
    retriever = None

# Initialize optimized Llama inference model with your explicit, working API key
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0.1,
    groq_api_key="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"
)

# Anti-hallucination prompt matching assignment requirements
qa_system_prompt = (
    "You are an advanced enterprise knowledge assistant. Synthesize a comprehensive answer "
    "based ONLY on the provided document context fragments.\n\n"
    "Rules:\n"
    "1. Citations: If page numbers or source names are present in your context details, mention them clearly.\n"
    "2. Hallucination Guardrail: If you do not know the answer or if it is not explicitly mentioned "
    "in the given context, state exactly: 'I am sorry, but the provided documentation does not contain information regarding this topic.'\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    ("human", "{input}"),
])

# 4. REST API Endpoint Layout
@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Core"])
async def query_knowledge_engine(payload: QueryInput, token: str = Security(api_key_header)):
    # Validate the security gateway credential token flag
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Enterprise Access Key Header")
        
    if not retriever:
        raise HTTPException(status_code=503, detail="Service Unavailable: Chroma Vector Storage is offline or missing local indices.")

    try:
        # Step A: Semantic document segment vector lookup retrieval stage
        docs = retriever.invoke(payload.question)
        context_string = "\n\n".join([doc.page_content for doc in docs])
        
        # Step B: Parse and isolate clean reference citation blocks
        citations_list = []
        for doc in docs:
            source_file = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
            page_num = doc.metadata.get('page', 0) + 1
            citations_list.append({
                "source": source_file,
                "page": page_num,
                "snippet_preview": doc.page_content[:120] + "..."
            })
            
        # Step C: Execute unified LCEL chain inference generation
        final_chain = qa_prompt | llm | StrOutputParser()
        answer = final_chain.invoke({
            "context": context_string,
            "input": payload.question
        })
        
        return {
            "query": payload.question,
            "answer": answer,
            "citations": citations_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Core Execution Fault: {str(e)}")

@app.get("/health", tags=["System Status"])
async def system_health_check():
    return {
        "status": "healthy" if db is not None else "degraded",
        "database_connected": db is not None,
        "engine_model": "llama-3.1-8b-instant"
    }