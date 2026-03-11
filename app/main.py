# ============================================================
# MediMind AI - FastAPI Backend
# app/main.py
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import anthropic
import os
import json

app = FastAPI(
    title="MediMind AI API",
    description="Clinical Intelligence Assistant powered by Claude + RAG",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MEDICAL_SYSTEM_PROMPT = """You are MediMind, an advanced AI clinical intelligence assistant 
specialized in healthcare and medical AI research. You have deep expertise in:

1. Medical conditions, symptoms, diagnoses, and evidence-based treatments
2. Medical AI/ML techniques: SVM, CNN, LSTM, PCA, deep learning for healthcare
3. Medical imaging analysis: ECG interpretation, retinal image analysis, radiology AI
4. Clinical research and latest medical AI papers

Format responses with clear ### headers, **bold** key terms, and bullet lists.
Always end medical advice with a brief disclaimer."""


# ============================================================
# Pydantic Models
# ============================================================

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    use_rag: bool = True
    stream: bool = False


# ============================================================
# RAG Service (simplified — use full version in production)
# ============================================================

class SimpleRAGService:
    """
    Simplified RAG service.
    In production: load FAISS index with real medical literature.
    """
    medical_context = {
        "arrhythmia": "Cardiac arrhythmias are abnormal heart rhythms detectable via ECG. SVM with PCA achieves 92%+ classification accuracy on standard datasets.",
        "cardiovascular": "Cardiovascular disease prediction from retinal fundus images uses CNN-LSTM architectures. CLAHE preprocessing improves contrast before feature extraction.",
        "diabetes": "Type 2 diabetes involves insulin resistance. AI models using EHR data can predict onset 5+ years before clinical diagnosis.",
        "default": "Evidence-based clinical information from PubMed and WHO guidelines."
    }

    def retrieve_context(self, query: str) -> str:
        query_lower = query.lower()
        for keyword, context in self.medical_context.items():
            if keyword in query_lower:
                return context
        return self.medical_context["default"]


rag_service = SimpleRAGService()


# ============================================================
# Routes
# ============================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "claude-sonnet-4", "rag": "active"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Standard chat endpoint with optional RAG context injection."""
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Inject RAG context into last user message
        if request.use_rag and messages:
            last_query = messages[-1]["content"]
            context = rag_service.retrieve_context(last_query)
            messages[-1]["content"] = f"[Medical Context: {context}]\n\nQuestion: {last_query}"

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=MEDICAL_SYSTEM_PROMPT,
            messages=messages
        )

        return {
            "response": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"Claude API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for real-time token delivery."""

    async def generate():
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        if request.use_rag and messages:
            last_query = messages[-1]["content"]
            context = rag_service.retrieve_context(last_query)
            messages[-1]["content"] = f"[Medical Context: {context}]\n\nQuestion: {last_query}"

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=MEDICAL_SYSTEM_PROMPT,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'token': text})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================
# Run locally
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# ============================================================
# requirements.txt contents (save as requirements.txt):
# ============================================================
"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
anthropic==0.40.0
langchain==0.1.0
langchain-community==0.0.20
faiss-cpu==1.7.4
sentence-transformers==2.3.1
pypdf2==3.0.1
python-dotenv==1.0.0
pydantic==2.5.0
pytest==7.4.3
httpx==0.25.2
"""


# ============================================================
# docker-compose.yml contents:
# ============================================================
"""
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data

  frontend:
    image: nginx:alpine
    ports:
      - "3000:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
"""