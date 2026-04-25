from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Model Configuration
    model_name: str = "allenai/DataDecide-dolma1_7-no-math-code-1B"

    # Training Configuration
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    fp16: bool = True

    # LoRA Configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Memory Configuration
    chroma_db_path: str = "./memory_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_memories: int = 3

    # Data Paths
    raw_data_path: str = "my_chats.txt"
    processed_data_path: str = "processed_chats.txt"
    checkpoint_dir: str = "./checkpoints"
    evolved_checkpoint_dir: str = "./evolved_checkpoints"
    preference_file: str = "training_signals.jsonl"

    # No-Code Filter
    forbidden_keywords: List[str] = field(default_factory=lambda: [
        "def ", "class ", "import ", "lambda", "return ", "if __name__",
        "std::", "iostream", "public static void", "System.out.println",
        "<html>", "<body>", "<div>", "<span>", "const ", "let ", "var "
    ])
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        r"```[\s\S]*?```", # Markdown code blocks
        r"`.*?`"           # Inline code
    ])

    # Server Configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # Scheduler Configuration
    evolve_interval_hours: float = 24.0
    auto_export_gguf: bool = False

    # Export Configuration
    merged_model_dir: str = "./merged_model"
    export_dir: str = "./exports"
    gguf_quantization: str = "Q4_K_M"
    llama_cpp_path: str = "./llama.cpp"

config = Config()
