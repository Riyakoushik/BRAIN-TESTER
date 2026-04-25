# Living Memory AI

A complete, self-evolving personal AI system designed to run on a Google Colab T4 GPU.

## Core Philosophy
This system is built to be a companion with persistent, "lossless" memory. It is trained exclusively on human conversational data and is strictly forbidden from generating or being trained on programming code.

## Architecture
The system consists of three interconnected loops:
1.  **Core Engine (Brain Stem):** A 1B-parameter Transformer model (`allenai/DataDecide-dolma1_7-no-math-code-1B`) fine-tuned via LoRA.
2.  **Infinite Memory (Hippocampus):** A persistent vector database (ChromaDB) that stores and retrieves past interactions and facts, injecting them into the model's context.
3.  **Self-Evolution (Plasticity):** A mechanism (`evolve.py`) that periodically fine-tunes the model on its own memories and user-selected preferred responses.

## File Structure
- `config.py`: Centralized configuration and hyperparameters.
- `data_prep.py`: Cleans raw chat logs and enforces the "no-code" policy.
- `model_loader.py`: Handles model loading, 4-bit quantization, and LoRA application.
- `train.py`: Initial fine-tuning script.
- `memory_store.py`: Interface for ChromaDB.
- `memory_utils.py`: High-level memory management and context retrieval.
- `chat.py`: Interactive chat interface with "Interactive Learning Mode".
- `evolve.py`: The self-evolution loop for continuous learning.
- `server.py`: FastAPI HTTP API server for always-on usage.
- `scheduler.py`: Automatic self-evolution scheduling with hot-reload.
- `export_gguf.py`: GGUF model export for phone/PC deployment.
- `sync_memory.py`: Memory export/import for device sync.
- `Modelfile`: Ollama deployment configuration.

## Setup and Usage

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (Optimized for T4)

### Installation
```bash
pip install -r requirements.txt
```

### 1. Data Preparation
Place your raw chat logs in `my_chats.txt` (or use the generated synthetic data) and run:
```bash
python data_prep.py
```

### 2. Initial Training
Fine-tune the base model on your data:
```bash
python train.py
```

### 3. Interactive Chat
Start the chat system:
```bash
python chat.py
```
- Type 'learn' to enter **Interactive Learning Mode**, where you can pick the best out of multiple AI responses.

### 4. Self-Evolution
Periodically run the evolution script to allow the model to permanently learn from its experiences:
```bash
python evolve.py
```

### 5. API Server (Always-On Mode)
Run the model as an HTTP API server instead of CLI:
```bash
python server.py
```

Endpoints:
- `POST /chat` — Send `{"message": "..."}`, get `{"response": "..."}`
- `POST /learn` — Get 3 candidate responses
- `POST /learn/select` — Pick the best response `{"message": "...", "choice": 1}`
- `POST /evolve` — Trigger evolution (runs in background, hot-reloads when done)
- `GET /memories` — View all stored memories
- `GET /health` — Health check

### 6. Auto-Evolution Scheduler
Run automatic evolution on a schedule:
```bash
python scheduler.py --interval-hours 24
python scheduler.py --interval-hours 12 --auto-export-gguf
```

### 7. Export to GGUF (for Phone & PC)
Convert your trained model to GGUF format for use with Ollama, LM Studio, or phone apps:

**Prerequisites:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
cd ..
```

**Export:**
```bash
python export_gguf.py
python export_gguf.py --quantization Q5_K_M  # higher quality
python export_gguf.py --quantization Q4_0     # smaller file
```

**Run on PC with Ollama:**
```bash
ollama create brain-tester -f Modelfile
ollama run brain-tester
```

**Run on Phone:**
Copy `exports/brain-tester-Q4_K_M.gguf` (~600MB) to your phone and open with:
- Android: [llama.cpp Android](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.android) or MLC Chat
- iOS: MLC Chat or LLM Farm

### 8. Memory Sync
Export memories to transfer between devices or back up:
```bash
python sync_memory.py export --output my_memories.json
python sync_memory.py import --input my_memories.json
```

## Architecture (Full)

```
Phone/PC (GGUF via Ollama)  <──sync──>  Server (GPU, Colab/cloud)
        │                                      │
        └── offline inference ──────── evolve.py (auto-scheduled)
                                               │
                                        export_gguf.py
                                               │
                                        new GGUF pushed to device
```

The inference runs locally on your device (fast, offline). Learning/evolution happens on the GPU server. Periodically sync the updated GGUF back to your device.

## No-Code Policy
This system implements multiple layers of code filtering:
- **Pre-training:** Uses a base model trained without code data.
- **Data Prep:** Automatically strips code blocks and technical keywords from training data.
- **Inference:** Prepends a "no-code" system prompt and applies a post-generation filter to remove accidental code output.
