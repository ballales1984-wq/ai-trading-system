#!/usr/bin/env python3
import subprocess
import os
os.chdir(r'c:\ai-trading-system')
result = subprocess.run(['git', 'status'], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

