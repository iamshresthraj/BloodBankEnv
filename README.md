# 🩸 BloodBankEnv - OpenEnv Hackathon 2026

## Motivation & Real-World Impact
In India, blood shortages and mismatched transfusions cause significant loss of life. Concurrently, highly perishable blood stocks frequently expire in storage due to poor logistics. **BloodBankEnv** trains intelligent agents to solve this exact healthcare logistics problem: prioritizing critical emergencies, dealing with stochastic donation camp inputs, and ensuring strict type compatibility—all while minimizing wastage.

## Setup & Deployment
1. **Local Run**: `pip install -r requirements.txt` then `uvicorn bloodbank.server:app --reload`
2. **Docker / HF Space**: 
   ```bash
   docker build -t bloodbankenv .
   docker run -p 8000:8000 bloodbankenv
   ```
   *Note: This repository is fully compliant with Hugging Face Space deployments and successfully passes standard `/reset` pings.*

## Tasks and Difficulty
1. **Easy:** `Basic Compatibility & Fulfillment` - Agent matches baseline requests without type mismatches.
2. **Medium:** `Expiry-Aware Stock Rotation` - Agent handles expirations, prioritizing older blood units first to prevent waste.
3. **Hard:** `Adaptive Management Under Uncertainty` - Agent manages incoming random donation drives, balances inventory across 8 types, and saves lives in emergency cases. 

## Evaluation
Run the pre-validation script locally:
```bash
python inference.py
```
Output will strictly follow:
`[START] task=...`
`[STEP] step=...`
`[END] success=true/false ...` 
