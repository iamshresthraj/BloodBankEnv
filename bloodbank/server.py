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
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --bg-color: #05070a;
            --accent-red: #ff3344;
            --accent-red-glow: rgba(255, 51, 68, 0.4);
            --card-bg: rgba(255, 255, 255, 0.03);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #ffffff;
            --text-secondary: #94a3b8;
            --success-green: #10b981;
        }

        * {
            box-sizing: border-box;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Plus Jakarta Sans', sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(255, 51, 68, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(255, 51, 68, 0.03) 0%, transparent 40%);
        }

        .container {
            width: 100%;
            max-width: 1000px;
            padding: 60px 40px;
            border-radius: 32px;
            background: var(--card-bg);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--border-color);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            text-align: center;
            position: relative;
            z-index: 1;
        }

        .container::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(135deg, var(--border-color), transparent 50%, var(--border-color));
            border-radius: 34px;
            z-index: -1;
            pointer-events: none;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 10px 20px;
            border-radius: 100px;
            background: rgba(16, 185, 129, 0.08);
            color: var(--success-green);
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 40px;
            border: 1px solid rgba(16, 185, 129, 0.2);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success-green);
            border-radius: 50%;
            margin-right: 10px;
            box-shadow: 0 0 12px var(--success-green);
            animation: pulse-green 2s infinite;
        }

        @keyframes pulse-green {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.5); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }

        h1 {
            font-size: 4rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(to bottom right, #fff 30%, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -2px;
        }

        p.subtitle {
            font-size: 1.25rem;
            color: var(--text-secondary);
            margin-top: 10px;
            margin-bottom: 60px;
            font-weight: 400;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            margin-top: 20px;
        }

        .card {
            background: rgba(255, 255, 255, 0.02);
            padding: 32px 24px;
            border-radius: 24px;
            border: 1px solid var(--border-color);
            text-align: left;
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            background: rgba(255, 255, 255, 0.04);
            transform: translateY(-8px);
            border-color: rgba(255, 51, 68, 0.3);
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.4);
        }

        .card-icon {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: var(--accent-red);
        }

        .card h3 {
            color: var(--text-primary);
            font-size: 1.2rem;
            margin-top: 0;
            margin-bottom: 12px;
            font-weight: 700;
        }

        .card p {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.6;
            margin: 0;
        }

        .live-source {
            margin-top: 40px;
            padding: 24px;
            border-radius: 20px;
            background: linear-gradient(90deg, rgba(255, 51, 68, 0.05), transparent);
            border: 1px solid rgba(255, 51, 68, 0.1);
            text-align: left;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .live-source-info h4 {
            margin: 0;
            font-size: 0.9rem;
            color: var(--accent-red);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .live-source-info p {
            margin: 4px 0 0 0;
            font-size: 1.1rem;
            color: var(--text-primary);
            font-weight: 500;
        }

        .btn-group {
            margin-top: 60px;
            display: flex;
            gap: 20px;
            justify-content: center;
        }

        .btn {
            padding: 16px 32px;
            border-radius: 100px;
            font-weight: 700;
            text-decoration: none;
            font-size: 0.95rem;
            letter-spacing: 0.5px;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }

        .btn-primary {
            background: var(--accent-red);
            color: white;
            box-shadow: 0 10px 25px -5px var(--accent-red-glow);
        }

        .btn-primary:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 35px -5px var(--accent-red-glow);
            filter: brightness(1.1);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: scale(1.05);
        }

        .api-badge {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            padding: 4px 8px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.05);
            font-size: 0.75rem;
            color: #60a5fa;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            h1 { font-size: 2.5rem; }
            .container { padding: 40px 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="status-badge">
            <div class="status-dot"></div>
            Environment API Online • Real-time
        </div>
        
        <h1>BloodBankEnv</h1>
        <p class="subtitle">An advanced RL simulation for strategic hospital blood bank management, integrated with real-time Indian blood stock data.</p>

        <div class="grid">
            <div class="card">
                <div class="card-icon">🏥</div>
                <h3>Dynamic State</h3>
                <p>Live inventory from 2000+ Indian blood banks via eRakt Kosh API integration.</p>
            </div>
            <div class="card">
                <div class="card-icon">⚖️</div>
                <h3>Intelligent Dispatch</h3>
                <p>Optimized allocation algorithms for life-critical priority matching and expiration control.</p>
            </div>
            <div class="card">
                <div class="card-icon">🏆</div>
                <h3>Grader Score</h3>
                <p>Performance based on fulfillment rate, minimal wastage, and emergency response speed.</p>
            </div>
        </div>

        <div class="live-source">
            <div class="live-source-info">
                <h4>Primary Data Integrity</h4>
                <p>eRakt Kosh (MoHFW, Govt. of India)</p>
            </div>
            <div class="api-badge">HTTPS / JSON / REAL-TIME</div>
        </div>

        <div class="btn-group">
            <a href="https://github.com/iamshresthraj/BloodBankEnv" target="_blank" class="btn btn-primary">
                View Repository
            </a>
            <a href="/docs" target="_blank" class="btn btn-secondary">
                API Specification
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
