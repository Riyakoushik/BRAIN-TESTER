"""
sync_memory.py — Export and import interactions as portable JSON.

Usage:
  python sync_memory.py export                     # export to memories_export.json
  python sync_memory.py export --output backup.json
  python sync_memory.py import --input backup.json  # import interactions
"""
import argparse
import json
import os
from memory_store import MemoryStore
from config import config


def export_memories(output_path="memories_export.json"):
    """Export all interactions to a JSON file."""
    store = MemoryStore()
    memories = store.get_all_memories()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"memories": memories, "count": len(memories)}, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(memories)} interactions to {output_path}")
    return output_path


def import_memories(input_path="memories_export.json"):
    """Import interactions from a JSON file."""
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    store = MemoryStore()
    imported = 0

    for mem in data.get("memories", []):
        text = mem.get("text", "")
        metadata = mem.get("metadata", {"type": "imported"})
        if text.strip():
            store.add_memory(text, metadata=metadata)
            imported += 1

    print(f"Imported {imported} interactions from {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory sync tool")
    parser.add_argument("action", choices=["export", "import"], help="Export or import interactions")
    parser.add_argument("--output", default="memories_export.json", help="Output file for export")
    parser.add_argument("--input", default="memories_export.json", help="Input file for import")
    args = parser.parse_args()

    if args.action == "export":
        export_memories(args.output)
    elif args.action == "import":
        import_memories(args.input)
