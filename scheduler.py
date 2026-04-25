"""
scheduler.py — Automatic self-evolution scheduler.

Can be run standalone or imported and started alongside the server.

Usage:
  python scheduler.py                          # standalone, evolve every 24h
  python scheduler.py --interval-hours 12      # evolve every 12h
  python scheduler.py --auto-export-gguf       # also re-export GGUF after each evolution
"""
import argparse
import time
import threading
import os
from config import config


class EvolutionScheduler:
    def __init__(self, interval_hours=24, auto_export_gguf=False, on_evolve_complete=None):
        self.interval = interval_hours * 3600  # convert to seconds
        self.auto_export = auto_export_gguf
        self.callback = on_evolve_complete  # e.g., chat_system.reload_adapters
        self._running = False
        self._thread = None
        self._evolve_count = 0

    def _loop(self):
        print(f"[Scheduler] Auto-evolution loop started. Interval: {self.interval / 3600}h")
        while self._running:
            time.sleep(self.interval)
            if not self._running:
                break

            print("[Scheduler] Starting scheduled evolution...")
            try:
                from evolve import evolve
                evolve()
                self._evolve_count += 1
                print(f"[Scheduler] Evolution #{self._evolve_count} complete.")

                # Hot-reload adapters if callback provided
                if self.callback:
                    self.callback()
                    print("[Scheduler] Adapters hot-reloaded.")

                # Optionally re-export GGUF
                if self.auto_export:
                    try:
                        from export_gguf import merge_lora, convert_to_gguf
                        adapter_path = config.evolved_checkpoint_dir
                        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                            merged = merge_lora(adapter_path)
                            convert_to_gguf(merged)
                            print("[Scheduler] GGUF re-exported.")
                        else:
                            print("[Scheduler] No evolved adapters found, skipping GGUF export.")
                    except Exception as e:
                        print(f"[Scheduler] GGUF export failed: {e}")

            except Exception as e:
                print(f"[Scheduler] Evolution failed: {e}")

    def start(self):
        if self._running:
            print("[Scheduler] Already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[Scheduler] Stopped.")

    def status(self):
        return {
            "running": self._running,
            "evolve_count": self._evolve_count,
            "interval_hours": self.interval / 3600
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-evolution scheduler")
    parser.add_argument("--interval-hours", type=float, default=24,
                        help="Hours between evolution cycles (default: 24)")
    parser.add_argument("--auto-export-gguf", action="store_true",
                        help="Re-export GGUF after each evolution")
    args = parser.parse_args()

    scheduler = EvolutionScheduler(
        interval_hours=args.interval_hours,
        auto_export_gguf=args.auto_export_gguf
    )
    scheduler.start()

    print("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.stop()
