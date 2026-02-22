@echo off
cd /d c:\ai-trading-system
git add README.md docs/TECHNICAL_DOCUMENTATION.md DEMO_RELEASE_CHECKLIST.md app/api/mock_data.py
git commit -m "Add technical docs, demo release checklist, and mock data API"
git push origin main
pause

