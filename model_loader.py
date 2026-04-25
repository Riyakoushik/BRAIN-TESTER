import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import config

def load_model_and_tokenizer(model_name=config.model_name, use_lora=True):
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {model_name}...")

    has_gpu = torch.cuda.is_available()
    device_map = "auto" if has_gpu else None

    # 4-bit quantization for GPU (fits 1B model in ~1.5GB VRAM)
    bnb_config = None
    if has_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float32 if not has_gpu else None,
        trust_remote_code=True
    )

    if use_lora:
        print("Applying LoRA...")
        if has_gpu:
            model = prepare_model_for_kbit_training(model)

        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer

if __name__ == "__main__":
    pass
