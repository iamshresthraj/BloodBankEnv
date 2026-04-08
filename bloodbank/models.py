from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class BloodType(str, Enum):
    O_NEG = "O-"
    O_POS = "O+"
    A_NEG = "A-"
    A_POS = "A+"
    B_NEG = "B-"
    B_POS = "B+"
    AB_NEG = "AB-"
    AB_POS = "AB+"

class Priority(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"

class BloodUnit(BaseModel):
    id: str
    blood_type: BloodType
    days_to_expiry: int

class Request(BaseModel):
    request_id: str
    blood_type: BloodType
    units_needed: int
    priority: Priority
    days_waiting: int = 0

class Observation(BaseModel):
    inventory: Dict[str, List[Dict[str, int]]]  # e.g., "O+": [{"days_to_expiry": 3, "count": 10}]
    pending_requests: List[Request]
    new_donations: Dict[str, int]  # Inflows from random donation camps today
    current_day: int
    total_mismatches_so_far: int
    total_wasted_units_so_far: int
    data_source: Optional[str] = None  # e.g. "eRakt Kosh - Delhi (15 blood banks)"
    is_live_data: bool = False

class Allocation(BaseModel):
    request_id: str
    allocated_units: int  # Number of units allocated
    prioritize_near_expiry: bool = True # Strategy flag for the agent
    allocated_blood_type: Optional[BloodType] = None # The specific blood type to dispense

class Action(BaseModel):
    allocations: List[Allocation]

class Reward(BaseModel):
    value: float
    metrics: Dict[str, float]

class State(BaseModel):
    task_id: str
    episode_id: str
    step_count: int
    is_done: bool
    score: float  # [0.0, 1.0] calculated by the specific task grader
    data_source: Optional[str] = None  # Attribution for live data origin
