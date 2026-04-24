import torch
from torch.utils.data import Dataset
from config import config

class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=config.max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        conversations = content.split("\n\n---\n\n")
        for conv in conversations:
            if not conv.strip():
                continue

            # Format: "User: ... AI: ..."
            # We want to train the model to predict the AI's response
            tokenized = tokenizer(
                conv,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )

            self.examples.append({
                "input_ids": tokenized["input_ids"].squeeze(),
                "attention_mask": tokenized["attention_mask"].squeeze(),
                "labels": tokenized["input_ids"].squeeze()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
