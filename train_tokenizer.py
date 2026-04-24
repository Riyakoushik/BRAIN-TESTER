from transformers import AutoTokenizer
from config import config
import os

def check_tokenizer():
    print(f"Verifying tokenizer for {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    test_text = "Hello, this is a test of the no-code tokenizer."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")

    save_path = "./tokenizer/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    check_tokenizer()
