import re
from config import config

def clean_text(text):
    # Remove markdown code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`.*?`", "", text)

    # Check for forbidden keywords and remove lines containing them
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if any(keyword in line for keyword in config.forbidden_keywords):
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()

def process_data(input_path, output_path):
    print(f"Processing data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple split by conversations if they are separated by dashes as in synthetic_data.py
    conversations = content.split("-" * 10)
    cleaned_conversations = []

    for conv in conversations:
        cleaned = clean_text(conv)
        if cleaned:
            cleaned_conversations.append(cleaned)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n---\n\n".join(cleaned_conversations))

    print(f"Data processing complete. Saved to {output_path}")

if __name__ == "__main__":
    process_data(config.raw_data_path, config.processed_data_path)
