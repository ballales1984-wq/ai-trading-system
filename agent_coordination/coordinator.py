"""
Enhanced Agent Coordination System
Multi-agent task coordination via shared files for two Kilo Code agents
"""
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

BASE = Path(__file__).parent
TASKS = BASE / "tasks"
PROGRESS = BASE / "in_progress"
DONE = BASE / "completed"
LOCKS = BASE / "locks"
COMMUNICATION = BASE / "communication"

# Create directories
for d in [TASKS, PROGRESS, DONE, LOCKS, COMMUNICATION]:
    d.mkdir(exist_ok=True)


class TaskPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    title: str
    description: str
    priority: str
    status: str
    created_by: str
    created_at: str
    claimed_by: Optional[str] = None
    started_at: Optional[str] = None
    completed_by: Optional[str] = None
    completed_at: Optional[str] = None
    files: List[str] = None
    depends_on: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.files is None:
            self.files = []
        if self.depends_on is None:
            self.depends_on = []
        if self.tags is None:
            self.tags = []


@dataclass
class AgentMessage:
    id: str
    from_agent: str
    to_agent: str
    message_type: str
    content: str
    timestamp: str
    related_task_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FileLock:
    """Manages file-level locks to prevent conflicts between agents"""
    
    @staticmethod
    def get_lock_file(filepath: str) -> Path:
        safe_name = hashlib.md5(filepath.encode()).hexdigest()[:8]
        return LOCKS / f"{safe_name}.lock"
    
    @staticmethod
    def acquire(filepath: str, agent_name: str, reason: str = "") -> bool:
        lock_file = FileLock.get_lock_file(filepath)
        if lock_file.exists():
            lock_data = json.loads(lock_file.read_text())
            if lock_data.get("agent") != agent_name:
                return False  # Already locked by another agent
        
        lock_data = {
            "filepath": filepath,
            "agent": agent_name,
            "reason": reason,
            "acquired_at": datetime.now().isoformat()
        }
        lock_file.write_text(json.dumps(lock_data, indent=2))
        return True
    
    @staticmethod
    def release(filepath: str, agent_name: str) -> bool:
        lock_file = FileLock.get_lock_file(filepath)
        if not lock_file.exists():
            return True
        
        lock_data = json.loads(lock_file.read_text())
        if lock_data.get("agent") != agent_name:
            return False  # Lock held by another agent
        
        lock_file.unlink()
        return True
    
    @staticmethod
    def is_locked(filepath: str) -> Optional[Dict]:
        lock_file = FileLock.get_lock_file(filepath)
        if lock_file.exists():
            return json.loads(lock_file.read_text())
        return None


class Coordinator:
    """Main coordinator class for multi-agent task management"""
    
    def __init__(self, name: str, other_agent_name: str = "Agent2"):
        self.name = name
        self.other_agent = other_agent_name
    
    def add(self, title: str, desc: str, priority: str = "medium", 
            files: List[str] = None, depends_on: List[str] = None,
            tags: List[str] = None) -> str:
        """Add a new task to the queue"""
        tid = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "id": tid,
            "title": title,
            "description": desc,
            "priority": priority,
            "status": "pending",
            "created_by": self.name,
            "created_at": datetime.now().isoformat(),
            "files": files or [],
            "depends_on": depends_on or [],
            "tags": tags or []
        }
        (TASKS / f"{tid}.json").write_text(json.dumps(data, indent=2))
        return tid
    
    def list_pending(self) -> List[tuple]:
        """List all pending tasks"""
        return [(f.stem, json.loads(f.read_text())) for f in TASKS.glob("*.json")]
    
    def list_by_agent(self, agent_name: str) -> List[tuple]:
        """List tasks claimed by a specific agent"""
        result = []
        for folder, status in [(PROGRESS, "in_progress"), (DONE, "completed")]:
            for f in folder.glob("*.json"):
                data = json.loads(f.read_text())
                if data.get("claimed_by") == agent_name or data.get("completed_by") == agent_name:
                    result.append((f.stem, data))
        return result
    
    def claim(self, tid: str) -> bool:
        """Claim a task for this agent"""
        f = TASKS / f"{tid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            
            # Check if any dependent tasks are completed
            if d.get("depends_on"):
                for dep_id in d["depends_on"]:
                    dep_file = DONE / f"{dep_id}.json"
                    if not dep_file.exists():
                        return False  # Dependency not completed
            
            d["status"] = "in_progress"
            d["claimed_by"] = self.name
            d["started_at"] = datetime.now().isoformat()
            (PROGRESS / f"{tid}.json").write_text(json.dumps(d, indent=2))
            f.unlink()
            return True
        return False
    
    def complete(self, tid: str) -> bool:
        """Mark a task as completed"""
        f = PROGRESS / f"{tid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            d["status"] = "completed"
            d["completed_by"] = self.name
            d["completed_at"] = datetime.now().isoformat()
            (DONE / f"{tid}.json").write_text(json.dumps(d, indent=2))
            f.unlink()
            return True
        return False
    
    def block(self, tid: str, reason: str) -> bool:
        """Block a task with a reason"""
        f = TASKS / f"{tid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            d["status"] = "blocked"
            d["blocked_reason"] = reason
            d["blocked_by"] = self.name
            (TASKS / f"{tid}.json").write_text(json.dumps(d, indent=2))
            return True
        return False
    
    def status(self) -> Dict:
        """Get overall status of all tasks"""
        return {
            "pending": len(list(TASKS.glob("*.json"))),
            "in_progress": len(list(PROGRESS.glob("*.json"))),
            "completed": len(list(DONE.glob("*.json"))),
            "my_tasks": len([f for f in PROGRESS.glob("*.json") 
                           if json.loads(f.read_text()).get("claimed_by") == self.name])
        }
    
    def send_message(self, to_agent: str, message_type: str, content: str,
                    related_task_id: str = None, metadata: Dict = None) -> str:
        """Send a message to another agent"""
        msg_id = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        msg = {
            "id": msg_id,
            "from_agent": self.name,
            "to_agent": to_agent,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "related_task_id": related_task_id,
            "metadata": metadata or {}
        }
        (COMMUNICATION / f"{msg_id}.json").write_text(json.dumps(msg, indent=2))
        return msg_id
    
    def read_messages(self) -> List[Dict]:
        """Read messages for this agent"""
        messages = []
        for f in COMMUNICATION.glob("*.json"):
            msg = json.loads(f.read_text())
            if msg["to_agent"] == self.name or msg["to_agent"] == "all":
                messages.append(msg)
        return messages
    
    def check_file_conflicts(self, files: List[str]) -> List[Dict]:
        """Check if any files are locked by the other agent"""
        conflicts = []
        for filepath in files:
            lock_info = FileLock.is_locked(filepath)
            if lock_info and lock_info.get("agent") != self.name:
                conflicts.append(lock_info)
        return conflicts


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: coordinator.py <agent_name> <command> [args...]")
        print("Commands: status, add, claim, complete, list, messages, lock, unlock")
        sys.exit(1)
    
    agent_name = sys.argv[1]
    cmd = sys.argv[2]
    other_agent = "Agent2" if agent_name == "Agent1" else "Agent1"
    
    c = Coordinator(agent_name, other_agent)
    
    if cmd == "status":
        print(json.dumps(c.status(), indent=2))
    elif cmd == "add":
        title = sys.argv[3] if len(sys.argv) > 3 else "New Task"
        desc = sys.argv[4] if len(sys.argv) > 4 else ""
        priority = sys.argv[5] if len(sys.argv) > 5 else "medium"
        print(f"Task created: {c.add(title, desc, priority)}")
    elif cmd == "list":
        for tid, task in c.list_pending():
            print(f"{tid}: {task['title']} ({task['status']}) - {task.get('description', '')}")
    elif cmd == "claim" and len(sys.argv) > 3:
        print(f"Claimed: {c.claim(sys.argv[3])}")
    elif cmd == "complete" and len(sys.argv) > 3:
        print(f"Completed: {c.complete(sys.argv[3])}")
    elif cmd == "messages":
        for msg in c.read_messages():
            print(f"[{msg['from_agent']} -> {msg['to_agent']}] {msg['message_type']}: {msg['content']}")
    elif cmd == "lock" and len(sys.argv) > 3:
        reason = sys.argv[4] if len(sys.argv) > 4 else ""
        print(f"Locked: {c.check_file_conflicts([sys.argv[3]])}")
    else:
        print(f"Unknown command: {cmd}")
