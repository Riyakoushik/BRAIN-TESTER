"""
server.py — Always-on FastAPI server for Living Memory AI.

Run with:
  python server.py
  # or: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import os
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from chat import ChatSystem
from config import config
import uvicorn

app = FastAPI(title="Living Memory AI", version="2.0")

# Global state
chat_system: Optional[ChatSystem] = None
_learn_cache: dict = {}  # cache learn responses by message hash


@app.on_event("startup")
def startup():
    global chat_system
    print("Loading model for server...")
    chat_system = ChatSystem()
    print("Server ready.")


# --- Request/Response models ---

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class LearnResponse(BaseModel):
    options: List[str]

class LearnSelectRequest(BaseModel):
    message: str
    choice: int  # 1, 2, or 3

class LearnSelectResponse(BaseModel):
    saved: str

class EvolveResponse(BaseModel):
    status: str

class MemoryResponse(BaseModel):
    memories: List[str]
    count: int


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": chat_system is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not chat_system:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    responses = chat_system.generate_response(req.message)
    ai_msg = responses[0]

    # Save to memory
    chat_system.orchestrator.store_interaction(req.message, ai_msg)

    return ChatResponse(response=ai_msg)


@app.post("/learn", response_model=LearnResponse)
def learn(req: ChatRequest):
    if not chat_system:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    responses = chat_system.generate_response(req.message, num_return_sequences=3)

    # Cache for later selection
    _learn_cache[req.message] = responses

    return LearnResponse(options=responses)


@app.post("/learn/select", response_model=LearnSelectResponse)
def learn_select(req: LearnSelectRequest):
    if not chat_system:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if req.choice < 1 or req.choice > 3:
        raise HTTPException(status_code=400, detail="Choice must be 1, 2, or 3")

    # Use cached responses if available, otherwise regenerate
    if req.message in _learn_cache:
        responses = _learn_cache.pop(req.message)
    else:
        responses = chat_system.generate_response(req.message, num_return_sequences=3)

    if req.choice > len(responses):
        raise HTTPException(status_code=400, detail=f"Only {len(responses)} options available")

    best = responses[req.choice - 1]
    chat_system.save_preference(req.message, best)
    chat_system.orchestrator.store_interaction(req.message, best, high_priority=True)

    return LearnSelectResponse(saved=best)


@app.post("/evolve", response_model=EvolveResponse)
def trigger_evolve():
    """Trigger self-evolution. This runs training in a background thread."""
    if not chat_system:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    def _evolve_and_reload():
        from evolve import evolve
        evolve()
        chat_system.reload_adapters()
        print("Evolution complete. Adapters reloaded.")

    thread = threading.Thread(target=_evolve_and_reload, daemon=True)
    thread.start()

    return EvolveResponse(status="Evolution started in background. Adapters will hot-reload when done.")


@app.get("/memories", response_model=MemoryResponse)
def get_memories():
    """Retrieve all stored memories."""
    store = chat_system.orchestrator.store
    all_mems = store.collection.get()
    docs = all_mems.get("documents", [])
    return MemoryResponse(memories=docs, count=len(docs))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
