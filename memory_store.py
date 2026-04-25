"""
memory_store.py — JSONL-based interaction storage.

All memories are stored as JSONL lines and get trained into the model
during evolution. No external database needed.
"""
import json
import os
from config import config


class MemoryStore:
    def __init__(self, file_path=config.interactions_file):
        self.file_path = file_path

    def add_memory(self, text, metadata=None):
        entry = {
            "text": text,
            "metadata": metadata or {"type": "conversation"}
        }
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_all_memories(self):
        if not os.path.exists(self.file_path):
            return []
        memories = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    memories.append(json.loads(line))
        return memories

    def count(self):
        if not os.path.exists(self.file_path):
            return 0
        count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


if __name__ == "__main__":
    store = MemoryStore(file_path="./test_memories.jsonl")
    store.add_memory("I love walking in the park during autumn.")
    print(f"Stored {store.count()} memories")
    print(f"All: {store.get_all_memories()}")
