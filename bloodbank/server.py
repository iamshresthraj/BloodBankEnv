from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .environment import BloodBankEnv
from .models import Action

app = FastAPI(title="BloodBank OpenEnv")
envs = {}

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BloodBankEnv | Interactive Dashboard</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Fira+Code:wght@400;500&display=swap');
        
        :root {
            --bg-color: #0b0f14;
            --panel-bg: rgba(20, 25, 33, 0.7);
            --card-bg: rgba(30, 36, 46, 0.8);
            --text-primary: #f2f5f8;
            --text-secondary: #a0aab2;
            --blood-red: #e63946;
            --neon-red: #ff4b4b;
            --neon-pink: #ff758f;
            --success: #2ecc71;
            --border-color: rgba(255, 255, 255, 0.08);
            --emergency: #ff3333;
            --urgent: #ff9933;
            --routine: #3399ff;
        }

        * { box-sizing: border-box; }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px;
            background-image: 
                radial-gradient(circle at 15% 10%, rgba(220, 20, 60, 0.1), transparent 30%),
                radial-gradient(circle at 85% 80%, rgba(220, 20, 60, 0.08), transparent 30%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
        }

        .header h1 {
            margin: 0;
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--neon-red), var(--neon-pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        select {
            background: var(--card-bg);
            color: white;
            border: 1px solid var(--border-color);
            padding: 10px;
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            cursor: pointer;
        }

        button {
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: 'Inter', sans-serif;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--blood-red), var(--neon-pink));
            color: white;
            box-shadow: 0 4px 15px rgba(230, 57, 70, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(230, 57, 70, 0.5);
        }

        .btn-secondary {
            background: var(--card-bg);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            flex-grow: 1;
        }

        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            backdrop-filter: blur(10px);
        }

        .panel h2 {
            margin-top: 0;
            font-size: 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }

        .badge {
            font-size: 0.8rem;
            padding: 3px 8px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.1);
        }

        .metrics-row {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 10px 15px;
            border-radius: 8px;
            flex-grow: 1;
            text-align: center;
        }

        .metric-card.score { border-color: var(--success); }
        .metric-card.penalty { border-color: var(--blood-red); }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 800;
            margin-top: 5px;
        }

        .state-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            overflow-y: auto;
            max-height: 50vh;
        }

        .section-box {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }

        .section-box h3 {
            margin-top: 0;
            font-size: 1rem;
            color: var(--text-secondary);
            margin-bottom: 10px;
        }

        .req-item, .inv-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 8px;
            border-left: 3px solid var(--text-secondary);
            font-size: 0.9rem;
        }

        .req-emergency { border-left-color: var(--emergency); }
        .req-urgent { border-left-color: var(--urgent); }
        .req-routine { border-left-color: var(--routine); }
        
        .inv-expiring { border-left-color: var(--blood-red); }

        textarea {
            width: 100%;
            height: 250px;
            background: #0f1218;
            color: #61dafb;
            font-family: 'Fira Code', monospace;
            border: 1px solid var(--border-color);
            padding: 15px;
            border-radius: 8px;
            resize: none;
            outline: none;
        }

        textarea:focus {
            border-color: var(--neon-pink);
            box-shadow: 0 0 10px rgba(255, 117, 143, 0.2);
        }

        .console-log {
            background: #000;
            color: #fff;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            margin-top: 15px;
        }
        
        .log-entry { margin-bottom: 5px; }
        .log-error { color: var(--blood-red); }
        .log-success { color: var(--success); }
        .log-info { color: #61dafb; }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-color); }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); }

        .hidden { display: none; }
        
        #overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.8); z-index: 999;
            display: flex; justify-content: center; align-items: center;
            color: white; font-size: 1.5rem; font-weight: bold;
            backdrop-filter: blur(5px);
        }
    </style>
</head>
<body>

    <div id="overlay" class="hidden">Processing...</div>

    <div class="header">
        <h1>🩸 BloodBankEnv</h1>
        <div class="controls">
            <select id="taskSelect">
                <option value="task_1_easy_basic_fulfillment">Task 1: Easy (Basic Fulfillment)</option>
                <option value="task_2_medium_expiry_rotation">Task 2: Medium (Expiry Rotation)</option>
                <option value="task_3_hard_adaptive_management" selected>Task 3: Hard (Adaptive Management)</option>
            </select>
            <button class="btn btn-secondary" onclick="resetEnv()">Reset / Start</button>
        </div>
    </div>

    <div class="main-grid">
        <!-- LEFT PANEL: State View -->
        <div class="panel">
            <h2>🌍 Environment State <span class="badge" id="dayBadge">Step 0</span></h2>
            
            <div class="metrics-row">
                <div class="metric-card score">
                    <div style="font-size: 0.8rem; color: #888;">Score</div>
                    <div class="metric-value" id="scoreVal">0.000</div>
                </div>
                <div class="metric-card penalty">
                    <div style="font-size: 0.8rem; color: #888;">Mismatches</div>
                    <div class="metric-value" id="mismatchVal">0</div>
                </div>
                <div class="metric-card penalty">
                    <div style="font-size: 0.8rem; color: #888;">Wasted Units</div>
                    <div class="metric-value" id="wasteVal">0</div>
                </div>
            </div>

            <div class="state-container">
                <div class="section-box">
                    <h3>🏥 Pending Requests</h3>
                    <div id="requestsList" style="font-size: 0.9rem; color: #a0aab2;"><i>Run reset to fetch requests...</i></div>
                </div>
                <div>
                    <div class="section-box">
                        <h3>📦 Inventory Stocks</h3>
                        <div id="inventoryList" style="font-size: 0.9rem; color: #a0aab2;"><i>Run reset to view inventory...</i></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT PANEL: Action Input -->
        <div class="panel">
            <h2>🤖 Artificial Intelligence Action</h2>
            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0; margin-bottom: 15px;">Instruct the agent by providing JSON distributions below.</p>
            
            <textarea id="actionJson">{
  "allocations": [
    {
      "request_id": "REQ_X",
      "allocated_units": 1,
      "prioritize_near_expiry": true
    }
  ]
}</textarea>
            
            <div style="margin-top: 15px;">
                <button class="btn btn-primary" style="width: 100%; border-radius: 8px;" onclick="stepEnv()" id="stepBtn" disabled>▶ Send Step Action</button>
            </div>

            <div class="console-log" id="consoleLog">
                <div class="log-info">> Waiting for initialization via Reset...</div>
            </div>
        </div>
    </div>

    <script>
        let currentEpisodeId = null;

        function log(message, type = 'info') {
            const consoleEl = document.getElementById('consoleLog');
            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.innerText = `> ${message}`;
            consoleEl.appendChild(entry);
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }

        async function resetEnv() {
            setLoading(true);
            const task_id = document.getElementById('taskSelect').value;
            try {
                const res = await fetch('/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task_id })
                });
                const data = await res.json();
                
                if (!res.ok) throw new Error(JSON.stringify(data));
                
                currentEpisodeId = data.state.episode_id;
                log(`Environment reset successfully. Episode ID: ${currentEpisodeId}`, 'success');
                
                updateUI(data.observation, data.state);
                
                document.getElementById('stepBtn').disabled = false;
                document.getElementById('actionJson').value = '{\n  "allocations": [\n  ]\n}';
                
            } catch (err) {
                console.error("ResetEnv Error:", err);
                log(`Reset Failed: ${err.message}`, 'error');
            } finally {
                setLoading(false);
            }
        }

        async function stepEnv() {
            if (!currentEpisodeId) return alert('Initialize first!');
            
            setLoading(true);
            const actionText = document.getElementById('actionJson').value;
            let actionData;
            try {
                actionData = JSON.parse(actionText);
            } catch (e) {
                setLoading(false);
                return log('Invalid JSON provided.', 'error');
            }

            try {
                const res = await fetch('/step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        episode_id: currentEpisodeId,
                        action: actionData
                    })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(JSON.stringify(data));
                
                const rewardVal = data.reward.value;
                const done = data.done;
                
                log(`Step executed. Reward: ${rewardVal.toFixed(2)}${done ? ' [EPISODE DONE]' : ''}`, rewardVal < 0 ? 'error' : 'success');
                
                updateUI(data.observation, data.state);

                if (done) {
                    document.getElementById('stepBtn').disabled = true;
                    log(`Episode Completed. Final Grader Score: ${data.state.score.toFixed(3)}`, 'success');
                }
            } catch (err) {
                console.error("StepEnv Error:", err);
                log(`Step Failed: ${err.message}`, 'error');
            } finally {
                setLoading(false);
            }
        }

        function updateUI(obs, state) {
            document.getElementById('scoreVal').innerText = state.score.toFixed(3);
            document.getElementById('mismatchVal').innerText = obs.total_mismatches_so_far;
            document.getElementById('wasteVal').innerText = obs.total_wasted_units_so_far;
            document.getElementById('dayBadge').innerText = `Step ${obs.current_day}`;

            const reqList = document.getElementById('requestsList');
            reqList.innerHTML = '';
            if (obs.pending_requests.length === 0) {
                reqList.innerHTML = '<i>No pending requests.</i>';
            } else {
                obs.pending_requests.forEach(req => {
                    const cls = `req-${String(req.priority).toLowerCase()}`;
                    reqList.innerHTML += `<div class="req-item ${cls}">
                        <strong>${req.request_id}</strong> - <b style="color:white">${req.blood_type}</b>
                        <br>Need: ${req.units_needed} units | Priority: ${req.priority}
                        <br><span style="font-size:0.8rem; color:#888;">Wait: ${req.days_waiting} days</span>
                    </div>`;
                });
            }

            const invList = document.getElementById('inventoryList');
            invList.innerHTML = '';
            let totalItems = 0;
            for (const [btype, batches] of Object.entries(obs.inventory)) {
                batches.forEach(b => {
                    totalItems++;
                    const expClass = b.days_to_expiry <= 3 ? 'inv-expiring' : '';
                    invList.innerHTML += `<div class="inv-item ${expClass}">
                        <strong>${btype}</strong> - ${b.count} units
                        <br><span style="font-size:0.8rem; color:#888;">Expires in ${b.days_to_expiry} days</span>
                    </div>`;
                });
            }
            if (totalItems === 0) {
                invList.innerHTML = '<i>Inventory is empty.</i>';
            }
        }

        function setLoading(isLoading) {
            const overlay = document.getElementById('overlay');
            if (isLoading) overlay.classList.remove('hidden');
            else overlay.classList.add('hidden');
        }
    </script>
</body>
</html>
"""

class ResetRequest(BaseModel):
    task_id: str

class StepRequest(BaseModel):
    episode_id: str
    action: Action

@app.post("/reset")
def reset(req: ResetRequest):
    env = BloodBankEnv(task_id=req.task_id)
    obs = env.reset()
    envs[env.episode_id] = env
    state = env.state()
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "state": state.model_dump() if hasattr(state, "model_dump") else state.dict()
    }

@app.post("/step")
def step(req: StepRequest):
    env = envs.get(req.episode_id)
    if not env:
        raise HTTPException(status_code=404, detail="Episode not found")
        
    obs, reward, done, info = env.step(req.action)
    state = env.state()
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "reward": reward.model_dump() if hasattr(reward, "model_dump") else reward.dict(),
        "done": done,
        "state": state.model_dump() if hasattr(state, "model_dump") else state.dict()
    }

@app.get("/state/{episode_id}")
def get_state(episode_id: str):
    env = envs.get(episode_id)
    if not env:
        raise HTTPException(status_code=404, detail="Episode not found")
    state = env.state()
    return state.model_dump() if hasattr(state, "model_dump") else state.dict()

@app.get("/")
def read_root():
    return HTMLResponse(content=HTML_CONTENT)
