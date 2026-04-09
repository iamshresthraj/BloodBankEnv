import uuid
import random
from typing import Tuple, Dict, Any, List, Optional
from .models import Action, Observation, State, Reward, Request, BloodType, Priority
from .data_fetcher import fetch_live_inventory, compute_live_distribution

class BloodBankEnv:
    # Data source attribution
    DATA_SOURCE = "eRakt Kosh, Ministry of Health & Family Welfare, Govt. of India"
    DATA_SOURCE_URL = "https://eraktkosh.mohfw.gov.in"

    def __init__(self, task_id: str = "task_1_easy_basic_fulfillment"):
        self.task_id = task_id
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.current_day = 1
        self.max_steps = 33
        self.is_done = False
        
        # Default: Indian Subcontinent Blood Type Probabilities
        # Overridden with live data from eRakt Kosh on reset()
        self.type_dist = {
            BloodType.O_POS: 0.37, BloodType.B_POS: 0.32, 
            BloodType.A_POS: 0.22, BloodType.AB_POS: 0.08,
            BloodType.O_NEG: 0.005, BloodType.B_NEG: 0.002, 
            BloodType.A_NEG: 0.002, BloodType.AB_NEG: 0.001
        }
        
        # State tracking
        self.inventory: Dict[BloodType, List[int]] = {bt: [] for bt in BloodType}
        self.requests: List[Request] = []
        
        # Live data source metadata (set during reset)
        self.data_source_state: Optional[str] = None
        self.data_source_banks: List[str] = []
        self.is_live_data: bool = False
        
        # Metrics for graders
        self.total_requests = 0
        self.fulfilled_requests = 0
        self.mismatches = 0
        self.wasted_units = 0
        self.total_donated_units = 0
        self.emergency_fulfilled = 0
        self.emergency_total = 0
        self.near_expiry_used = 0
        self.total_used = 0

    def generate_random_type(self) -> BloodType:
        r = random.random()
        cumulative = 0.0
        for bt, prob in self.type_dist.items():
            cumulative += prob
            if r <= cumulative:
                return bt
        return BloodType.O_POS

    def _seed_from_live_data(self) -> bool:
        """
        Fetch real-time blood stock from eRakt Kosh and seed inventory.
        Returns True if live data was used, False otherwise.
        """
        try:
            live_stock, state_name, bank_names = fetch_live_inventory()
            
            # Update blood type distribution from real data
            self.type_dist = compute_live_distribution(live_stock)
            self.data_source_state = state_name
            self.data_source_banks = bank_names
            self.is_live_data = True
            
            # Seed inventory from real stock counts
            # Each unit gets a random expiry between 3-35 days
            for bt, count in live_stock.items():
                for _ in range(count):
                    self.inventory[bt].append(random.randint(3, 35))
                    self.total_donated_units += 1
            
            print(
                f"[ENV] Live data loaded from eRakt Kosh | "
                f"State: {state_name} | "
                f"Blood Banks: {len(bank_names)} | "
                f"Total Units: {sum(live_stock.values())}",
                flush=True
            )
            return True
            
        except Exception as e:
            print(f"[ENV] eRakt Kosh unavailable ({e}), using synthetic data", flush=True)
            return False

    def _seed_synthetic_data(self):
        """Fallback: seed inventory with random synthetic data."""
        self.is_live_data = False
        self.data_source_state = None
        self.data_source_banks = []
        for bt in BloodType:
            for _ in range(random.randint(5, 20)):
                self.inventory[bt].append(random.randint(5, 30))
                self.total_donated_units += 1

    def reset(self) -> Observation:
        self.__init__(self.task_id)
        
        # Always use synthetic data first for instant response
        # This ensures the /reset endpoint never times out
        self._seed_synthetic_data()
        
        # Then try to upgrade to live data (non-blocking, best-effort)
        try:
            import threading
            def _try_live():
                try:
                    live_stock, state_name, bank_names = fetch_live_inventory()
                    self.type_dist = compute_live_distribution(live_stock)
                    self.data_source_state = state_name
                    self.data_source_banks = bank_names
                    self.is_live_data = True
                    print(
                        f"[ENV] Live data loaded in background from eRakt Kosh | "
                        f"State: {state_name} | "
                        f"Blood Banks: {len(bank_names)}",
                        flush=True
                    )
                except Exception as e:
                    print(f"[ENV] Background live data fetch failed: {e}", flush=True)
            t = threading.Thread(target=_try_live, daemon=True)
            t.start()
        except Exception:
            pass
                
        return self._get_observation()

    def _get_observation(self, donations: Dict[str, int] = None) -> Observation:
        inv_summary = {}
        for bt, expiries in self.inventory.items():
            counts = {}
            for e in expiries:
                counts[e] = counts.get(e, 0) + 1
            inv_summary[bt.value] = [{"days_to_expiry": d, "count": c} for d, c in counts.items()]
            
        # Build data source label
        ds_label = None
        if self.is_live_data and self.data_source_state:
            ds_label = f"eRakt Kosh - {self.data_source_state} ({len(self.data_source_banks)} blood banks)"

        return Observation(
            inventory=inv_summary,
            pending_requests=self.requests,
            new_donations=donations or {},
            current_day=self.current_day,
            total_mismatches_so_far=self.mismatches,
            total_wasted_units_so_far=self.wasted_units,
            data_source=ds_label,
            is_live_data=self.is_live_data
        )

    def is_compatible(self, requested: BloodType, allocated: BloodType) -> bool:
        # Full Transfusion Matrix
        compatibility = {
            BloodType.O_NEG: [BloodType.O_NEG],
            BloodType.O_POS: [BloodType.O_NEG, BloodType.O_POS],
            BloodType.A_NEG: [BloodType.O_NEG, BloodType.A_NEG],
            BloodType.A_POS: [BloodType.O_NEG, BloodType.O_POS, BloodType.A_NEG, BloodType.A_POS],
            BloodType.B_NEG: [BloodType.O_NEG, BloodType.B_NEG],
            BloodType.B_POS: [BloodType.O_NEG, BloodType.O_POS, BloodType.B_NEG, BloodType.B_POS],
            BloodType.AB_NEG: [BloodType.O_NEG, BloodType.A_NEG, BloodType.B_NEG, BloodType.AB_NEG],
            BloodType.AB_POS: list(BloodType)
        }
        return allocated in compatibility.get(requested, [])

    # Maximum total reward across the entire episode
    MAX_TOTAL_REWARD = 100.0

    @property
    def max_step_reward(self) -> float:
        return self.MAX_TOTAL_REWARD / self.max_steps  # ~3.33 per step

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # --- 100 Point System ---
        # Agent starts each step with the full step budget (~3.33 pts).
        # Penalties deduct from it. The step reward is clamped to [0, max_step_reward].
        step_budget = self.max_step_reward
        deductions = 0.0

        # 1. Process Allocations
        allocations_made = 0
        for alloc in action.allocations:
            req = next((r for r in self.requests if r.request_id == alloc.request_id), None)
            if not req:
                continue
                
            # If agent didn't specify alternate type, assume they meant exact match
            alloc_type = alloc.allocated_blood_type if alloc.allocated_blood_type else req.blood_type
            
            available = len(self.inventory[alloc_type])
            units_to_take = min(alloc.allocated_units, available)
            
            if units_to_take > 0:
                # Sort inventory by expiry to prioritize older units
                self.inventory[alloc_type].sort()
                
                # Expiry Rotator Tracker
                if self.inventory[alloc_type][0] <= 5: 
                    self.near_expiry_used += units_to_take
                self.total_used += units_to_take

                # Match logic - penalize if agent forces wrong types
                if not self.is_compatible(req.blood_type, alloc_type):
                    self.mismatches += units_to_take
                    deductions += 2.0  # Heavy deduction for mismatch
                    # Do not dispense lethal mismatch to patient; leaves inventory intact but penalizes.
                else:
                    self.inventory[alloc_type] = self.inventory[alloc_type][units_to_take:]
                    req.units_needed -= units_to_take
                    allocations_made += units_to_take

                    if req.priority == Priority.EMERGENCY:
                        self.emergency_fulfilled += 1
                
                if req.units_needed <= 0:
                    self.fulfilled_requests += 1
                    if req in self.requests:
                        self.requests.remove(req)

        # Deduct if no allocations were made at all (idle penalty)
        if allocations_made == 0 and len(self.requests) > 0:
            deductions += 1.0

        # 2. Advance time: age inventory, track wastage
        for bt in list(self.inventory.keys()):
            new_inv = []
            for d in self.inventory[bt]:
                if d - 1 <= 0:
                    self.wasted_units += 1
                    deductions += 0.3  # Deduction per wasted unit
                else:
                    new_inv.append(d - 1)
            self.inventory[bt] = new_inv

        # 3. Increase waiting time for unfulfilled requests, penalize delays
        for req in self.requests:
            req.days_waiting += 1
            if req.priority == Priority.EMERGENCY:
                deductions += 0.5  # Heavier deduction for delayed emergencies
            elif req.priority == Priority.URGENT:
                deductions += 0.2
            else:
                deductions += 0.1

        # 4. Stochastic events: Generate new patient requests
        num_new_reqs = random.randint(1, 4)
        for _ in range(num_new_reqs):
            self.total_requests += 1
            priority = random.choice([Priority.ROUTINE, Priority.URGENT, Priority.EMERGENCY])
            if priority == Priority.EMERGENCY:
                self.emergency_total += 1
                
            self.requests.append(Request(
                request_id=f"REQ_{self.current_day}_{uuid.uuid4().hex[:4]}",
                blood_type=self.generate_random_type(),
                units_needed=random.randint(1, 4),
                priority=priority
            ))

        # 5. Stochastic events: Donation Camp (5-15% chance)
        donations = {}
        if random.random() < 0.15:
            for _ in range(random.randint(5, 15)):
                bt = self.generate_random_type()
                self.inventory[bt].append(random.randint(25, 35))
                donations[bt.value] = donations.get(bt.value, 0) + 1
                self.total_donated_units += 1

        self.current_day += 1
        if self.step_count >= self.max_steps:
            self.is_done = True
        
        # Clamp step reward: never negative, never above step budget
        reward_val = max(0.0, step_budget - deductions)
        reward_val = min(reward_val, step_budget)
            
        obs = self._get_observation(donations)
        reward = Reward(value=reward_val, metrics={
            "fulfilled": self.fulfilled_requests,
            "max_step_reward": self.max_step_reward
        })
        
        return obs, reward, self.is_done, {}

    def get_grader_score(self) -> float:
        """The specific grader logic based on the task_id"""
        if self.total_requests == 0: return 0.001
        
        fulfillment_rate = self.fulfilled_requests / max(self.total_requests, 1)
        mismatch_penalty = 1.0 if self.mismatches > 0 else 0.0
        
        if self.task_id == "task_1_easy_basic_fulfillment":
            # Grader 1: Fulfill requests and avoid killing patients.
            score = (fulfillment_rate * 1.0) - mismatch_penalty
            
        elif self.task_id == "task_2_medium_expiry_rotation":
            # Grader 2: Must fulfill, but also utilize near-expiry items and minimize waste.
            waste_rate = self.wasted_units / max(self.total_donated_units, 1)
            expiry_utilization = self.near_expiry_used / max(self.total_used, 1)
            
            score = (fulfillment_rate * 0.5) + (expiry_utilization * 0.3) - (waste_rate * 0.2) - mismatch_penalty
            
        else: # task_3_hard_adaptive_management
            # Grader 3: Emergency prioritization is paramount, low waste, balanced matching over stochastic inflows.
            emergency_rate = self.emergency_fulfilled / max(self.emergency_total, 1)
            waste_rate = self.wasted_units / max(self.total_donated_units, 1)
            
            score = (emergency_rate * 0.6) + (fulfillment_rate * 0.3) - (waste_rate * 0.1) - mismatch_penalty

        return max(0.001, min(0.999, score)) # Normalize strictly within (0, 1)

    def state(self) -> State:
        ds_label = None
        if self.is_live_data and self.data_source_state:
            ds_label = f"eRakt Kosh - {self.data_source_state} ({len(self.data_source_banks)} blood banks)"
        return State(
            task_id=self.task_id,
            episode_id=self.episode_id,
            step_count=self.step_count,
            is_done=self.is_done,
            score=self.get_grader_score(),
            data_source=ds_label
        )
