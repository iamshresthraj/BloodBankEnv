---
title: BloodBankEnv
emoji: 🩸
colorFrom: red
colorTo: pink
sdk: docker
app_port: 8000
---

# 🩸 BloodBankEnv - OpenEnv Hackathon 2026

## 🌟 Overview & Motivation
In many parts of the world, specifically in the Indian Subcontinent, blood shortages and mismatched transfusions cause significant loss of life. Concurrently, highly perishable blood stocks frequently expire in storage due to poor logistics and lack of rotation. 

**BloodBankEnv** is an OpenEnv-compliant reinforcement learning (RL) and LLM environment designed to train intelligent agents to solve this exact healthcare logistics problem. Agents must balance competing priorities naturally:
- Prioritizing critical emergencies and urgent patient requests.
- Dealing with stochastic, unpredictable donation camp inflows across 8 different blood types.
- Ensuring strict type compatibility to avoid life-fatal mismatches.
- Implementing FIFO (First-In-First-Out) rotations to minimize blood expiration and wastage.

## 🚀 Setup & Deployment

### Local Run 
```bash
pip install -r requirements.txt
uvicorn bloodbank.server:app --reload
```

### Docker / Hugging Face Space (OpenEnv Compliant)
This repository is pre-configured and fully compliant with standard Hugging Face Space OpenEnv deployment, natively supporting the `/reset` and `/step` OpenEnv webhook structures.
```bash
docker build -t bloodbankenv .
docker run -p 8000:8000 -e HF_TOKEN="your_token" bloodbankenv
```

## 🧠 Environment Mechanics

### State (Observation)
At each step, the agent observes:
- `inventory`: A dictionary of blood types (e.g., `O+`, `AB-`) containing arrays of units and their `days_to_expiry`.
- `pending_requests`: An array of hospital requests, each with a `request_id`, `blood_type`, `units_needed`, `priority` (Routine, Urgent, Emergency), and `days_waiting`.
- `new_donations`: A breakdown of randomized inflows from donation camps that occurred today.
- `current_day`: The current day loop of the simulation.
- Performance tracking variables (`total_mismatches_so_far`, `total_wasted_units_so_far`).

### Action Space (Allocations)
The AI agent must return a strict JSON representing its dispatch allocations for the day:
```json
{
  "allocations": [
    {
      "request_id": "REQ_1_a1b2",
      "allocated_units": 2,
      "prioritize_near_expiry": true
    }
  ]
}
```

### Rewards & Penalties
- **+2.0** per unit successfully matched and distributed.
- **+10.0** bonus for fast emergency handling.
- **-1.0 to -10.0** for request delays (scaled based on priority level).
- **-2.0** for every blood unit wasted (expired before use).
- **-50.0** Severe penalty for blood type mismatch (life-critical safety error).

## 🏆 Tracks and Difficulty Grading
1. **Easy:** `task_1_easy_basic_fulfillment`
   - Goal: Basic fulfillment and type adherence.
   - Evaluation: Agent matches baseline requests without type mismatches. Grader focuses purely on fulfillment ratios.
2. **Medium:** `task_2_medium_expiry_rotation`
   - Goal: Expiry-Aware Stock Rotation.
   - Evaluation: Agent must fulfill requests while actively preventing expirations. Grader scores highly based on active `expiry_utilization` and minimizing the overall `waste_rate`.
3. **Hard:** `task_3_hard_adaptive_management`
   - Goal: Adaptive Management Under Uncertainty.
   - Evaluation: Agent manages incoming random donation drives, balances long-term inventory across all 8 types, and saves lives in emergency cases. Grader heavily weights the `emergency_rate` alongside strict waste minimization.

## 🧪 Evaluation Script
Run the pre-validation inference script locally using default bounds or the target HF Router parameters:
```bash
export HF_TOKEN="hf_your_token_here"
python inference.py
```
Output will strictly follow OpenEnv stdout grading conventions:
`[START] task=...`
`[STEP] step=...`
`[END] success=true/false ...` 

---
*Built for the 2026 Meta PyTorch Hackathon.*
