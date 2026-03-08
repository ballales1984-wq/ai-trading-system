import subprocess
import os

os.chdir('c:/ai-trading-system')
result = subprocess.run(['git', 'checkout', 'HEAD', '--', 'frontend/src/pages/Dashboard.tsx'], 
                       capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
