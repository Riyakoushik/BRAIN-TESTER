import os
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from model_loader import load_model_and_tokenizer
from dataset import ChatDataset
from config import config

def train():
    model, tokenizer = load_model_and_tokenizer()

    dataset = ChatDataset(config.processed_data_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none", # Disable wandb/etc for simplicity
        remove_unused_columns=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {config.checkpoint_dir}")
    model.save_pretrained(config.checkpoint_dir)
    tokenizer.save_pretrained(config.checkpoint_dir)

if __name__ == "__main__":
    # Check if we should process data first
    import os
    if not os.path.exists(config.processed_data_path):
        from data_prep import process_data
        process_data(config.raw_data_path, config.processed_data_path)

    train()
