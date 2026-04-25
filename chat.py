import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import config
from memory_utils import MemoryOrchestrator
import json
import re
import os

class ChatSystem:
    def __init__(self, model_path=config.checkpoint_dir):
        self._model_lock = threading.Lock()
        self.orchestrator = MemoryOrchestrator()

        print(f"Loading model from {model_path}...")
        from model_loader import load_model_and_tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(use_lora=False)

        # Load best available adapters
        if os.path.exists(os.path.join(config.evolved_checkpoint_dir, "adapter_config.json")):
            print("Loading evolved LoRA adapters...")
            self.model = PeftModel.from_pretrained(self.model, config.evolved_checkpoint_dir)
        elif os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(self.model, model_path)

        self.system_prompt = (
            "You are a helpful, human-like AI assistant. "
            "You must NEVER output any programming code, markdown code blocks, or technical scripts. "
            "Your goal is to be a companion with persistent memory. "
            "Be warm, thoughtful, and conversational.\n"
        )

    def filter_output(self, text):
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`.*?`", "", text)
        for keyword in config.forbidden_keywords:
            if keyword in text:
                text = text.replace(keyword, "[REMOVED]")
        return text

    def generate_response(self, user_input, num_return_sequences=1):
        prompt = f"{self.system_prompt}\nUser: {user_input}\nAI:"

        with self._model_lock, torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id
            )

        responses = []
        for output in outputs:
            full_text = self.tokenizer.decode(output, skip_special_tokens=True)
            ai_part = full_text.split("AI:")[-1].strip()
            responses.append(self.filter_output(ai_part))

        return responses

    def save_preference(self, user_input, preferred_response):
        signal = {
            "prompt": user_input,
            "preferred": preferred_response
        }
        with open(config.preference_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal) + "\n")

    def reload_adapters(self, adapter_path=None):
        """Hot-reload LoRA adapters without restarting. Used after evolution."""
        path = adapter_path or config.evolved_checkpoint_dir
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            from model_loader import load_model_and_tokenizer
            new_model, _ = load_model_and_tokenizer(use_lora=False)
            new_model = PeftModel.from_pretrained(new_model, path)
            new_model.eval()
            with self._model_lock:
                self.model = new_model
            print(f"Hot-reloaded adapters from {path}")
        else:
            print(f"No adapters found at {path}, keeping current model.")

    def run_chat(self):
        print("--- Living Memory AI Started ---")
        print("Type 'exit' to quit, 'learn' to enter interactive learning mode.")
        print(f"Interactions stored: {self.orchestrator.get_interaction_count()}")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                break

            if user_input.lower() == 'learn':
                user_msg = input("Enter message for learning mode: ")
                responses = self.generate_response(user_msg, num_return_sequences=3)
                print("\nOptions:")
                for i, res in enumerate(responses):
                    print(f"{i+1}: {res}")

                choice = input("\nPick the best response (1-3) or 'none': ")
                if choice.isdigit() and 1 <= int(choice) <= 3:
                    best_res = responses[int(choice)-1]
                    self.save_preference(user_msg, best_res)
                    self.orchestrator.store_interaction(user_msg, best_res, high_priority=True)
                    print("Preference saved! Will be trained into model on next evolution.")
                continue

            responses = self.generate_response(user_input)
            ai_msg = responses[0]
            print(f"AI: {ai_msg}")

            # Save interaction for future training
            self.orchestrator.store_interaction(user_input, ai_msg)

if __name__ == "__main__":
    chat = ChatSystem()
    chat.run_chat()
