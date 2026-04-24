from memory_store import MemoryStore
from config import config

class MemoryOrchestrator:
    def __init__(self):
        self.store = MemoryStore()

    def get_context(self, user_input):
        memories = self.store.query_memories(user_input)
        if not memories:
            return ""

        context = "\nRelevant past memories:\n"
        for i, mem in enumerate(memories):
            context += f"- {mem}\n"
        return context

    def store_interaction(self, user_msg, ai_msg, high_priority=False):
        # Store as a single memory entry
        memory_text = f"User said: {user_msg} | AI responded: {ai_msg}"
        metadata = {"type": "interaction", "priority": "high" if high_priority else "normal"}
        self.store.add_memory(memory_text, metadata=metadata)

    def add_fact(self, fact):
        self.store.add_memory(fact, metadata={"type": "fact"})

# Simple graph-like association helper (simulated for now)
def find_related_topics(text):
    # This could be expanded to use a real graph DB or NLP-based keyword extraction
    # For now, it's a placeholder for future 'lossless memory' expansion
    return []
