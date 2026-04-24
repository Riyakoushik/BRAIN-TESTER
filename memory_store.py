import chromadb
from chromadb.utils import embedding_functions
from config import config
import os

class MemoryStore:
    def __init__(self, db_path=config.chroma_db_path):
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )

        self.collection = self.client.get_or_create_collection(
            name="personal_memories",
            embedding_function=self.embedding_fn
        )

    def add_memory(self, text, metadata=None):
        # Generate a simple ID based on current collection size
        mem_id = f"mem_{self.collection.count()}"
        self.collection.add(
            documents=[text],
            metadatas=[metadata] if metadata else [{"type": "conversation"}],
            ids=[mem_id]
        )
        print(f"Added memory: {text[:50]}...")

    def query_memories(self, query_text, n_results=config.top_k_memories):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

if __name__ == "__main__":
    # Quick test
    store = MemoryStore(db_path="./test_memory_db")
    store.add_memory("I love walking in the park during autumn.")
    mems = store.query_memories("What do I like to do in autumn?")
    print(f"Retrieved: {mems}")
