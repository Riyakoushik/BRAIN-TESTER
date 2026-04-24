import json
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model_loader import load_model_and_tokenizer
from dataset import ChatDataset
from config import config
from memory_store import MemoryStore
import os

def prepare_evolution_data():
    store = MemoryStore()
    memories = store.collection.get()

    evolution_text = ""
    # Process regular memories
    if memories['documents']:
        for doc in memories['documents']:
            evolution_text += doc + "\n\n---\n\n"

    # Process high-priority preferences
    if os.path.exists(config.preference_file):
        with open(config.preference_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                evolution_text += f"User: {data['prompt']}\nAI: {data['preferred']}\n\n---\n\n"

    with open("evolution_data.txt", "w", encoding="utf-8") as f:
        f.write(evolution_text)

    return "evolution_data.txt"

from peft import PeftModel

def evolve():
    data_file = prepare_evolution_data()

    # Load model with existing LoRA weights if any
    model, tokenizer = load_model_and_tokenizer(use_lora=False)

    if os.path.exists(os.path.join(config.checkpoint_dir, "adapter_config.json")):
        print(f"Loading existing adapters from {config.checkpoint_dir}...")
        model = PeftModel.from_pretrained(model, config.checkpoint_dir, is_trainable=True)

    dataset = ChatDataset(data_file, tokenizer)

    training_args = TrainingArguments(
        output_dir=config.evolved_checkpoint_dir,
        num_train_epochs=1, # Evolve is typically shorter
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        fp16=config.fp16,
        logging_steps=5,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting self-evolution...")
    trainer.train()

    print(f"Saving evolved model to {config.evolved_checkpoint_dir}")
    model.save_pretrained(config.evolved_checkpoint_dir)

if __name__ == "__main__":
    evolve()
