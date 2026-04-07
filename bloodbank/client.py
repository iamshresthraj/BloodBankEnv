import os
import httpx
from pydantic import BaseModel
from .models import Action, Observation

class Result(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: dict = {}
    score: float = 0.0

class BloodBankEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    @classmethod
    async def from_docker_image(cls, image_name: str = None):
        # Factory method required by OpenEnv standard
        # In a real deployed test, this orchestrates the docker container.
        # Here we connect to the locally running HTTP server.
        env_url = os.getenv("ENV_API_URL", "http://localhost:8000")
        return cls(base_url=env_url)

    async def reset(self) -> Result:
        task_id = os.getenv("TASK_NAME", "task_3_hard_adaptive_management")
        async with httpx.AsyncClient() as http:
            resp = await http.post(f"{self.base_url}/reset", json={"task_id": task_id})
            data = resp.json()
            self.episode_id = data["state"]["episode_id"]
            return Result(
                observation=Observation(**data["observation"]),
                score=data["state"].get("score", 0.0)
            )

    async def step(self, action: Action) -> Result:
        async with httpx.AsyncClient() as http:
            resp = await http.post(f"{self.base_url}/step", json={
                "episode_id": getattr(self, "episode_id", "unknown"),
                "action": action.dict()
            })
            data = resp.json()
            return Result(
                observation=Observation(**data["observation"]),
                reward=data["reward"]["value"],
                done=data["done"],
                score=data["state"].get("score", 0.0)
            )
            
    async def close(self):
        # OpenEnv cleanup sequence
        pass
