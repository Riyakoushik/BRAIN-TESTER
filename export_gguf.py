"""
export_gguf.py — Merge LoRA adapters into base model and convert to GGUF.

Prerequisites:
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && make
"""
import argparse
import subprocess
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import config


def merge_lora(adapter_path):
    """Load base model in fp16 (NOT 4-bit), merge LoRA adapters, save merged model."""
    print(f"Loading base model {config.model_name} in fp16 for merging...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: Must load in full/half precision — cannot merge LoRA into quantized model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU to avoid VRAM issues during merge
        trust_remote_code=True
    )

    print(f"Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    merged_path = config.merged_model_dir
    os.makedirs(merged_path, exist_ok=True)
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")
    return merged_path


def convert_to_gguf(merged_path, llama_cpp_path="./llama.cpp", output_dir="./exports", quantization="Q4_K_M"):
    """Convert merged HF model to GGUF and quantize."""
    os.makedirs(output_dir, exist_ok=True)

    fp16_gguf = os.path.join(output_dir, "brain-tester-f16.gguf")
    quantized_gguf = os.path.join(output_dir, f"brain-tester-{quantization}.gguf")

    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_cpp_path, "llama-quantize")

    # Validate llama.cpp exists
    if not os.path.exists(convert_script):
        print(f"ERROR: {convert_script} not found.")
        print("Please clone and build llama.cpp first:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && make")
        return None

    # Step 1: Convert HF to GGUF (fp16)
    print("Converting to GGUF (fp16)...")
    subprocess.run(
        ["python", convert_script, merged_path, "--outfile", fp16_gguf, "--outtype", "f16"],
        check=True
    )

    # Step 2: Quantize
    if not os.path.exists(quantize_bin):
        print(f"WARNING: {quantize_bin} not found, skipping quantization. Use the fp16 GGUF.")
        return fp16_gguf

    print(f"Quantizing to {quantization}...")
    subprocess.run(
        [quantize_bin, fp16_gguf, quantized_gguf, quantization],
        check=True
    )

    # Cleanup fp16 intermediate
    os.remove(fp16_gguf)
    print(f"GGUF model exported to: {quantized_gguf}")
    return quantized_gguf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Living Memory AI to GGUF format")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to LoRA adapter dir (default: evolved_checkpoints or checkpoints)")
    parser.add_argument("--llama-cpp-path", default="./llama.cpp",
                        help="Path to llama.cpp repo clone")
    parser.add_argument("--output-dir", default="./exports",
                        help="Output directory for GGUF file")
    parser.add_argument("--quantization", default="Q4_K_M",
                        choices=["Q4_0", "Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
                        help="Quantization type")
    args = parser.parse_args()

    # Pick best available adapter path
    adapter_path = args.adapter_path
    if adapter_path is None:
        if os.path.exists(os.path.join(config.evolved_checkpoint_dir, "adapter_config.json")):
            adapter_path = config.evolved_checkpoint_dir
        elif os.path.exists(os.path.join(config.checkpoint_dir, "adapter_config.json")):
            adapter_path = config.checkpoint_dir
        else:
            print("ERROR: No trained adapters found. Run train.py first.")
            exit(1)

    print(f"Using adapters from: {adapter_path}")
    merged = merge_lora(adapter_path)
    gguf = convert_to_gguf(merged, args.llama_cpp_path, args.output_dir, args.quantization)

    if gguf:
        print(f"\nDone! To use with Ollama:")
        print(f"  ollama create brain-tester -f Modelfile")
        print(f"\nTo use on phone, copy {gguf} to your device and open with MLC Chat or llama.cpp Android.")
