"""
evolve.py — Self-evolution: trains the model on all stored interactions.

Every interaction from chat is saved to interactions.jsonl.
Running evolve() fine-tunes the model on all interactions + preferences,
permanently baking those memories into the model weights.
"""
import json
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model_loader import load_model_and_tokenizer
from dataset import ChatDataset
from config import config
from memory_store import MemoryStore
import os
from peft import PeftModel


def prepare_evolution_data():
    evolution_text = ""

    # Process all stored interactions
    store = MemoryStore()
    memories = store.get_all_memories()
    for mem in memories:
        text = mem.get("text", "")
        if text.strip():
            evolution_text += text + "\n\n---\n\n"

    # Process high-priority preferences
    if os.path.exists(config.preference_file):
        with open(config.preference_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    evolution_text += f"User: {data['prompt']}\nAI: {data['preferred']}\n\n---\n\n"

    if not evolution_text.strip():
        print("No interactions found. Chat with the AI first to build training data.")
        return None

    with open("evolution_data.txt", "w", encoding="utf-8") as f:
        f.write(evolution_text)

    return "evolution_data.txt"


def evolve():
    data_file = prepare_evolution_data()
    if data_file is None:
        return

    has_gpu = torch.cuda.is_available()

    # Load model with existing LoRA weights if any
    model, tokenizer = load_model_and_tokenizer(use_lora=False)

    if os.path.exists(os.path.join(config.checkpoint_dir, "adapter_config.json")):
        print(f"Loading existing adapters from {config.checkpoint_dir}...")
        model = PeftModel.from_pretrained(model, config.checkpoint_dir, is_trainable=True)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = ChatDataset(data_file, tokenizer)
    print(f"Evolution dataset: {len(dataset)} examples from interactions + preferences")

    training_args = TrainingArguments(
        output_dir=config.evolved_checkpoint_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        fp16=has_gpu and config.fp16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting self-evolution (training interactions into model weights)...")
    trainer.train()

    print(f"Saving evolved model to {config.evolved_checkpoint_dir}")
    model.save_pretrained(config.evolved_checkpoint_dir)
    print("Evolution complete. All interactions are now part of the model.")

if __name__ == "__main__":
    evolve()
