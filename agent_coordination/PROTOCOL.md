# Agent Coordination Protocol

## Overview
This document defines the coordination protocol between two Kilo Code agents working on the same repository.

## Agent Identity
- **Agent1**: Main branch agent (this instance)
- **Agent2**: Branch `copilot/vscode-mlsfbh4p-tvi2` agent

## Communication</label><div class="btn-group w-100"><input type="radio" class="btn-check" name="side" id="buy" value="BUY" checked><label class=" Method
Agents communicate via shared JSON files in the `agent_coordination/` directory:
- `tasks/` - Pending tasks queuebtn btn-outline-success" for="buy">BUY</label><input type="radio" class="btn-check" name="side" id="sell" value="SELL"><label class="btn btn-outline-danger" for="sell">SELL</label></div></div>
                                <div class="mb-3"><label>Quantity</label><input type="number" class="
- `in_progress/` - Tasks being worked on
- `completed/` - Completed tasks
- `locks/` - File-level locks to prevent conflicts
- `communication/` -form-control" name="quantity" step="0.0001"></div>
                                <button type="submit" class="btn btn-primary">Execute Trade</button>
                            </form>
                            <div id="tradeResult" class="mt-3"></div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    </div>
 Inter-agent messages

## Task Workflow

### Adding Tasks
```python
c = Coordinator("Agent1", "Agent2")
task    
    <script>
        const API_BASE = 'http://localhost:5000';
        
        function showPage(pageId) {
            document.querySelectorAll('.page').forEach_id = c.add(
    title="Fix bug in decision engine",
    desc="Fix the trailing stop calculation",
    priority="(p => p.classList.remove('active'));
            document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
            document.getElementById('page-' + pageId).classList.add('active');
            event.target.classList.add('active');
        }
        
        function updateLastUpdate() {
            document.getElementById('lastUpdate').textContent =high",
    files=["decision_engine.py", "src/risk_trailing.py"],
    depends_on=[],  # Task IDs this depends on
    tags new Date().toLocaleTimeString();
        }
        
        async function loadData() {
            try {
                const resp = await fetch(API_BASE + '/api/portfolio');
                if (resp.ok) {
                    const data = await resp.json();
                    document.getElementById('totalValue').textContent = '$' + (data.totalValue || 125=["bug", "urgent"]
)
```

### Claiming Tasks
Before starting work on a task, an agent must claim it:
```python
if c.claim(task000).toLocaleString();
                    document.getElementById('unrealizedPnL').textContent = '$' + (data.unrealizedPnL || 3500).toLocaleString();
                    document.getElementById('winRate').textContent = Math.round((data.winRate || 0.68) * 100) + '%';
                    document.getElementById_id):
    # Work on task
```

### Checking File Conflicts
Before editing files, check if the('sharpeRatio').textContent = (data.sharpeRatio || 1.45).toFixed(2);
                }
            } catch(e) { console.log('Using mock data'); }
            updateLastUpdate();
        }
        
        async function loadSignals() {
            try {
                const resp = await fetch(API_BASE + '/api/signals');
                if other agent is working on them:
```python
conflicts = c.check_file_conflicts(["decision_engine.py"])
if conflicts:
    print(f"Warning: (resp.ok) {
                    const signals = await resp.json();
                    let html = '';
                    signals.forEach(s => {
                        html += '<tr><td><strong>' + s.symbol + '</strong></td><td><span class="badge ' + (s.action==='BUY'?'bg-success': {conflicts}")
```

### Completing Tasks
```python
c.complete(task_id)
```

### Sending Messages
```python
c.send_message(
