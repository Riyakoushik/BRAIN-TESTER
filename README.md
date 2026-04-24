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

## No-Code Policy
This system implements multiple layers of code filtering:
- **Pre-training:** Uses a base model trained without code data.
- **Data Prep:** Automatically strips code blocks and technical keywords from training data.
- **Inference:** Prepends a "no-code" system prompt and applies a post-generation filter to remove accidental code output.
