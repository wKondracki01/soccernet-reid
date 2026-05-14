"""Tiny script for verifying launch_bg.py detach actually works on Windows.

Prints a heartbeat every second for 2 minutes. If launched via launch_bg.py
(--kind raw), should survive the launching shell's exit. Verify by polling
after the launcher returns.

Usage: see scripts/launch_bg.py docstring.
"""
import os
import sys
import time

print(f"detach_test PID={os.getpid()} starting", flush=True)
for i in range(120):
    print(f"  alive at {i}s", flush=True)
    time.sleep(1)
print("DONE", flush=True)
sys.exit(0)
