#!/usr/bin/env python3
import subprocess
import os

os.chdir(r'c:\ai-trading-system')

# Git add
result = subprocess.run(['git', 'add', '-A'], capture_output=True, text=True)
print("=== GIT ADD ===")
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Git status
result = subprocess.run(['git', 'status'], capture_output=True, text=True)
print("\n=== GIT STATUS ===")
print(result.stdout)
