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
        self.orchestrator = MemoryOrchestrator()

        print(f"Loading model from {model_path}...")
        # Load base model (in 4-bit if GPU available)
        from model_loader import load_model_and_tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(use_lora=False)

        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(self.model, model_path)

        self.system_prompt = (
            "You are a helpful, human-like AI assistant. "
            "You must NEVER output any programming code, markdown code blocks, or technical scripts. "
            "Your goal is to be a companion with persistent memory. "
            "Use the provided context from previous interactions to stay consistent.\n"
        )

    def filter_output(self, text):
        # Remove markdown code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code
        text = re.sub(r"`.*?`", "", text)
        # Remove common code keywords if they appear in a technical context
        for keyword in config.forbidden_keywords:
            if keyword in text:
                text = text.replace(keyword, "[REMOVED]")
        return text

    def generate_response(self, user_input, num_return_sequences=1):
        context = self.orchestrator.get_context(user_input)
        prompt = f"{self.system_prompt}\n{context}\nUser: {user_input}\nAI:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
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
            # Extract only the AI's part
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
            # Get the base model (unwrap PeftModel if already wrapped)
            base = self.model
            if hasattr(self.model, 'base_model'):
                base = self.model.base_model.model
            self.model = PeftModel.from_pretrained(base, path)
            self.model.eval()
            print(f"Hot-reloaded adapters from {path}")
        else:
            print(f"No adapters found at {path}, keeping current model.")

    def run_chat(self):
        print("--- Living Memory AI Started ---")
        print("Type 'exit' to quit, 'learn' to enter interactive learning mode.")

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
                    print("Preference saved!")
                continue

            responses = self.generate_response(user_input)
            ai_msg = responses[0]
            print(f"AI: {ai_msg}")

            # Save to memory
            self.orchestrator.store_interaction(user_input, ai_msg)

if __name__ == "__main__":
    # Note: This requires the model to be trained/available.
    chat = ChatSystem()
    chat.run_chat()
