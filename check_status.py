#!/usr/bin/env python3
import subprocess
import os

os.chdir(r'c:\ai-trading-system')
result = subprocess.run(['git', 'status'], capture_output=True, text=True)
print("=== GIT STATUS ===")
print(result.stdout)
if result.stderr:
    print("=== ERRORS ===")
    print(result.stderr)

result2 = subprocess.run(['git', 'log', '--oneline', '-5'], capture_output=True, text=True)
print("=== RECENT COMMITS ===")
print(result2.stdout)
