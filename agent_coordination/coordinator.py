"""
Agent Coordination System
Multi-agent task coordination via shared files
"""
import os, json
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
TASKS = BASE / "tasks"
PROGRESS = BASE / "in_progress"
DONE = BASE / "completed"

for d in [TASKS, PROGRESS, DONE]:
    d.mkdir(exist_ok=True)

class Coordinator:
    def __init__(self, name): self.name = name
    
    def add(self, title, desc, priority="medium"):
        tid = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {"id": tid, "title": title, "description": desc, 
                "priority": priority, "created_by": self.name,
                "created_at": datetime.now().isoformat(), "status": "pending"}
        (TASKS / f"{tid}.json").write_text(json.dumps(data, indent=2))
        return tid
    
    def list_pending(self):
        return [(f.stem, json.loads(f.read_text())) for f in TASKS.glob("*.json")]
    
    def claim(self, tid):
        f = TASKS / f"{tid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            d["status"] = "in_progress"
            d["claimed_by"] = self.name
            (PROGRESS / f"{tid}.json").write_text(json.dumps(d, indent=2))
            f.unlink()
            return True
        return False
    
    def complete(self, tid):
        f = PROGRESS / f"{tid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            d["status"] = "completed"
            d["completed_by"] = self.name
            (DONE / f"{tid}.json").write_text(json.dumps(d, indent=2))
            f.unlink()
            return True
        return False
    
    def status(self):
        return {
            "pending": len(list(TASKS.glob("*.json"))),
            "in_progress": len(list(PROGRESS.glob("*.json"))),
            "completed": len(list(DONE.glob("*.json")))
        }

if __name__ == "__main__":
    import sys
    c = Coordinator("Agent1")
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "status":
            print(c.status())
        elif cmd == "add" and len(sys.argv) > 2:
            print(c.add(sys.argv[2], sys.argv[3] if len(sys.argv)>3 else ""))
    else:
        print("Pending:", c.list_pending())
