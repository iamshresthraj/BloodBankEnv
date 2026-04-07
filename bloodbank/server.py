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
    <title>BloodBankEnv | OpenEnv Hackathon</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        :root {
            --bg-color: #0d0f12;
            --accent-glow: rgba(220, 20, 60, 0.6);
            --card-bg: rgba(255, 255, 255, 0.03);
            --text-primary: #f2f2f2;
            --text-secondary: #a0aab2;
            --blood-red: #e63946;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(220, 20, 60, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(220, 20, 60, 0.05), transparent 25%);
        }

        .container {
            max-width: 900px;
            padding: 40px;
            border-radius: 24px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            text-align: center;
            animation: fadeIn 1s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #ff4b4b, #ff9090);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin-bottom: 40px;
            font-weight: 300;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 8px 16px;
            border-radius: 50px;
            background: rgba(46, 204, 113, 0.1);
            color: #2ecc71;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 30px;
            border: 1px solid rgba(46, 204, 113, 0.2);
            box-shadow: 0 0 15px rgba(46, 204, 113, 0.2);
        }

        .status-dot {
            width: 10px;
            height: 10px;
            background-color: #2ecc71;
            border-radius: 50%;
            margin-right: 8px;
            box-shadow: 0 0 8px #2ecc71;
            animation: pulse 2s infinite ease-in-out;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(46, 204, 113, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .card {
            background: rgba(0, 0, 0, 0.2);
            padding: 25px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            text-align: left;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            border-color: rgba(220, 20, 60, 0.3);
        }

        .card h3 {
            color: var(--text-primary);
            font-size: 1.2rem;
            margin-top: 0;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 0;
        }

        .btn-group {
            margin-top: 40px;
            display: flex;
            gap: 15px;
            justify-content: center;
        }

        .btn {
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: var(--blood-red);
            color: white;
            box-shadow: 0 4px 15px rgba(230, 57, 70, 0.3);
        }

        .btn-primary:hover {
            background: #ff4b4b;
            box-shadow: 0 6px 20px rgba(230, 57, 70, 0.4);
            transform: scale(1.02);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: scale(1.02);
        }
        
        .code-block {
            background: #1a1c21;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            color: #61dafb;
            margin-top: 15px;
            font-size: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="status-badge">
            <div class="status-dot"></div>
            Environment API Online
        </div>
        
        <h1>🩸 BloodBankEnv</h1>
        <p class="subtitle">An OpenEnv RL Simulation for Emergency Blood Bank Logistics</p>

        <div class="grid">
            <div class="card">
                <h3>🏥 Environment State</h3>
                <p>Monitors blood type inventory, expiry rotations, patient requests (Routine to Emergency), and stochastic donation inflows.</p>
            </div>
            <div class="card">
                <h3>⚖️ Auto Allocations</h3>
                <p>LLM Agents allocate blood to patients while maintaining strict compatibility and preventing life-threatening mismatches.</p>
            </div>
            <div class="card">
                <h3>🏆 Action Rewards</h3>
                <p>Agents are graded on fulfillment rate, minimal blood expiration (wastage), and prioritized emergency response times.</p>
            </div>
        </div>

        <div style="text-align: left; margin-top: 30px;">
            <p style="color: var(--text-secondary); margin-bottom: 5px;"><strong>API Endpoints Active:</strong></p>
            <div class="code-block">
                POST /reset - Initialize episode state<br>
                POST /step - Submit agent observation actions<br>
                GET /state/{id} - Retrieve episodic grading state
            </div>
        </div>

        <div class="btn-group">
            <a href="https://github.com/iamshresthraj/BloodBankEnv" target="_blank" class="btn btn-primary">
                View Source
            </a>
            <a href="/docs" target="_blank" class="btn btn-secondary">
                View API Docs
            </a>
        </div>
    </div>
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
    return {"observation": obs.dict(), "state": state.dict()}

@app.post("/step")
def step(req: StepRequest):
    env = envs.get(req.episode_id)
    if not env:
        raise HTTPException(status_code=404, detail="Episode not found")
        
    obs, reward, done, info = env.step(req.action)
    state = env.state()
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "state": state.dict()
    }

@app.get("/state/{episode_id}")
def get_state(episode_id: str):
    env = envs.get(episode_id)
    if not env:
        raise HTTPException(status_code=404, detail="Episode not found")
    return env.state().dict()

@app.get("/")
def read_root():
    return HTMLResponse(content=HTML_CONTENT)
