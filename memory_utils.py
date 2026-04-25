from memory_store import MemoryStore
from config import config


class MemoryOrchestrator:
    def __init__(self):
        self.store = MemoryStore()

    def store_interaction(self, user_msg, ai_msg, high_priority=False):
        memory_text = f"User: {user_msg}\nAI: {ai_msg}"
        metadata = {"type": "interaction", "priority": "high" if high_priority else "normal"}
        self.store.add_memory(memory_text, metadata=metadata)

    def add_fact(self, fact):
        self.store.add_memory(fact, metadata={"type": "fact"})

    def get_interaction_count(self):
        return self.store.count()
