# Living Memory AI

A self-evolving personal AI that runs **100% locally** on your machine. All memories are trained directly into the model weights — no external database, no cloud, no data leaves your system.

## Core Philosophy
This system is built to be a companion with persistent memory baked into the model itself. Every conversation gets fine-tuned into the weights, so the model file IS your memory. Copy the model to any device and all memories come with it.

## Architecture
1. **Core Engine (Brain Stem):** A 1B-parameter Transformer model (`allenai/DataDecide-dolma1_7-no-math-code-1B`) fine-tuned via LoRA.
2. **In-Model Memory:** Conversations are stored as JSONL, then periodically trained into the model weights via `evolve.py`. The model itself becomes the memory.
3. **Self-Evolution (Plasticity):** Running `evolve.py` fine-tunes the model on all your interactions, permanently embedding them into the weights.

## Hardware Requirements
- **Minimum:** 3GB VRAM GPU + 8GB RAM (or CPU-only with 12GB+ RAM)
- **Recommended:** 4GB+ VRAM GPU + 12GB RAM
- Works on: Windows, Linux, macOS (with compatible GPU or CPU fallback)

## File Structure
- `config.py`: Centralized configuration and hyperparameters.
- `data_prep.py`: Cleans raw chat logs and enforces the "no-code" policy.
- `model_loader.py`: Handles model loading, 4-bit quantization, and LoRA application.
- `train.py`: Initial fine-tuning script.
- `memory_store.py`: JSONL-based interaction storage.
- `memory_utils.py`: Interaction orchestration.
- `chat.py`: Interactive chat interface with "Interactive Learning Mode".
- `evolve.py`: Self-evolution — trains all interactions into model weights.
- `server.py`: FastAPI HTTP API server for always-on usage.
- `scheduler.py`: Automatic self-evolution scheduling with hot-reload.
- `export_gguf.py`: GGUF model export for phone/PC deployment.
- `sync_memory.py`: Interaction export/import for backup.
- `Modelfile`: Ollama deployment configuration.

## Setup and Usage

### Prerequisites
- Python 3.8+
- GPU with 3GB+ VRAM (optional — CPU works too, just slower)

### Installation
```bash
pip install -r requirements.txt
```

**Note:** The base model (~2GB) downloads on first run. After that, everything is fully offline.

### 1. Data Preparation
Place your raw chat logs in `my_chats.txt` and run:
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
- Every conversation is automatically saved to `interactions.jsonl`
- Type `learn` to enter **Interactive Learning Mode** — pick the best out of 3 responses
- Type `exit` to quit

### 4. Self-Evolution (Daily Training)
Train all your conversations into the model weights:
```bash
python evolve.py
```
Run this daily (or whenever you want the model to absorb new interactions). After evolution, the model permanently remembers everything you've discussed.

### 5. API Server (Always-On Mode)
Run the model as an HTTP API server:
```bash
python server.py
```

Endpoints:
- `POST /chat` — Send `{"message": "..."}`, get `{"response": "..."}`
- `POST /learn` — Get 3 candidate responses
- `POST /learn/select` — Pick the best response `{"message": "...", "choice": 1}`
- `POST /evolve` — Trigger evolution (runs in background, hot-reloads when done)
- `GET /stats` — View interaction count
- `GET /health` — Health check

### 6. Auto-Evolution Scheduler
Run automatic evolution on a schedule:
```bash
python scheduler.py --interval-hours 24
python scheduler.py --interval-hours 12 --auto-export-gguf
```

### 7. Export to GGUF (for Phone & PC)
Convert your trained model (with all memories baked in) to GGUF format:

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

### 8. Backup & Restore Interactions
Export your interaction history (for backup or transfer):
```bash
python sync_memory.py export --output my_backup.json
python sync_memory.py import --input my_backup.json
```

## How Memory Works

```
Chat (interactions.jsonl) ──> evolve.py ──> Model Weights (permanent memory)
                                                │
                                          export_gguf.py
                                                │
                                          GGUF file (portable, runs anywhere)
```

1. You chat → interactions saved to `interactions.jsonl`
2. You run `evolve.py` → interactions trained into model weights
3. The model now **permanently remembers** those conversations
4. Export to GGUF → all memories travel with the model file

**No database. No cloud. No external dependencies. The model IS the memory.**

## No-Code Policy
This system implements multiple layers of code filtering:
- **Pre-training:** Uses a base model trained without code data.
- **Data Prep:** Automatically strips code blocks and technical keywords from training data.
- **Inference:** Prepends a "no-code" system prompt and applies a post-generation filter.
