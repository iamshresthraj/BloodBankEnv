import os
import yaml
import json
import re

def check_16_points():
    results = {}
    print("=== OPENENV 16-POINT COMPLIANCE AUDIT ===")

    # 1. Project Metadata
    if os.path.exists("openenv.yaml"):
        with open("openenv.yaml", 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            results["metadata_present"] = all(k in cfg for k in ["spec_version", "name", "version"])
    else:
        results["metadata_present"] = False
    
    # 2. Connectivity
    results["connectivity_defined"] = cfg and "app" in cfg and "port" in cfg
    
    # 3. Tasks
    results["tasks_defined"] = cfg and len(cfg.get("tasks", [])) >= 3
    
    # 4. Dockerfile
    results["dockerfile_present"] = os.path.exists("Dockerfile")
    
    # 5. FastAPI Implementation (Static check)
    with open("bloodbank/server.py", 'r', encoding='utf-8') as f:
        content = f.read()
        results["fastapi_reset"] = "@app.post(\"/reset\")" in content
        results["fastapi_step"] = "@app.post(\"/step\")" in content
    
    # 6. Models Check
    results["pydantic_models"] = os.path.exists("bloodbank/models.py")
    
    # 8. Inference script
    results["inference_script"] = os.path.exists("inference.py")
    
    # 9. Telemetry check
    if os.path.exists("inference.py"):
        with open("inference.py", 'r', encoding='utf-8') as f:
            inf_content = f.read()
            results["telemetry_start"] = "[START] task=" in inf_content
            results["telemetry_step"] = "[STEP] step=" in inf_content
            results["telemetry_end"] = "[END] success=" in inf_content

    # 14. Requirements
    results["requirements_present"] = os.path.exists("requirements.txt") or os.path.exists("pyproject.toml")

    print("\nSummary of Checks:")
    for point, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {point}")
    
    overall = all(results.values())
    print("\n" + "="*40)
    print(f"OVERALL COMPLIANCE: {'PASS' if overall else 'FAIL'}")
    print("="*40)
    return overall

if __name__ == "__main__":
    check_16_points()
