import math
import pandas as pd
from collections import deque
from src.utils import load_config

class Autoscaler:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        
        self.cfg = config
        self.capacity = config['server']['capacity']
        self.min_servers = config['server']['min_servers']
        self.max_servers = config['server']['max_servers']
        
        # State tracking
        self.current_servers = self.min_servers
        self.last_scale_time = -999
        
        # Stability Window
        self.window_size = config['stability']['window_minutes']
        self.scale_in_buffer = config['stability']['scale_in_buffer']
        self.history_demand = deque(maxlen=self.window_size)
        
    def get_safety_factor(self, hour):
        """Determine safety factor based on hourly volatility profile."""
        high_risk_hours = self.cfg['safety']['high_risk_hours']
        economic_hours = self.cfg['safety']['economic_hours']
        
        if hour in high_risk_hours:
            return self.cfg['safety']['factors']['high_risk']
        elif hour in economic_hours:
            return self.cfg['safety']['factors']['economic']
        else:
            return self.cfg['safety']['factors']['standard']

    def detect_ddos(self, current_load, forecast):
        multiplier = self.cfg['anomaly']['ddos_multiplier']
        # Avoid false positives on very low traffic
        if forecast > 10 and current_load > forecast * multiplier:
            return True
        return False

    def decide_scale(self, timestamp_minute, current_load, forecast, hour, sigma=0.0, cv=0.0):
        # 1. Anomaly Detection
        is_ddos = self.detect_ddos(current_load, forecast)
        
        # Default Safety Factor from Config (Baseline)
        safety_factor = self.get_safety_factor(hour)
        
        # --- GEN 2: ADAPTIVE SAFETY MARGIN ---
        # Logic: K factor depends on CV (Model Confidence)
        k_factor = 0.0
        if cv < 0.1:
            k_factor = 1.0 # High Confidence
        elif cv <= 0.3:
            k_factor = 1.2 # Normal
        else:
            k_factor = 1.35 # Low Confidence / Volatile -> High Safety
            
        # 2. Raw Demand Calculation
        if is_ddos:
            raw_demand = math.ceil(current_load / self.capacity)
            raw_demand = min(raw_demand, self.cfg['anomaly']['ddos_max_servers'])
            safety_factor = 1.0 
        else:
            # Hybrid Scaling with Confidence
            # Demand = (Forecast + K * Sigma) / Capacity
            # If sigma is 0 (first run), we fall back to safety_factor multiplier method
            
            if sigma > 0:
                adjusted_forecast = forecast + (k_factor * sigma)
                demand_pred = math.ceil(adjusted_forecast / self.capacity)
            else:
                # Fallback to Gen 1 logic
                demand_pred = math.ceil((forecast * safety_factor) / self.capacity)
                
            demand_react = math.ceil((current_load * safety_factor) / self.capacity)
            raw_demand = max(demand_pred, demand_react)
            
            # Circuit Breaker
            raw_demand = min(raw_demand, self.max_servers)

        # Ensure min servers
        raw_demand = max(raw_demand, self.min_servers)

        # 3. Update Stability Window
        self.history_demand.append(raw_demand)
        
        # 4. Stability Logic (Windowing + Hysteresis + Cooldown + Step Scaling)
        minutes_since = timestamp_minute - self.last_scale_time
        action = "HOLD"
        cooldown_out = self.cfg['cooldown']['scale_out_minutes']
        cooldown_in = self.cfg['cooldown']['scale_in_minutes']
        
        # New: Step Scaling Limit
        max_step = self.cfg['stability'].get('max_step_change', 5) 
        
        # Cold Start Check: Are we still warming up?
        is_warming_up = len(self.history_demand) < self.window_size
        
        target_servers = self.current_servers

        # DECISION TREE
        
        # Case A: Potential SCALE OUT
        # We use MIN of the window to ensure sustained high demand
        sustained_demand_up = max(self.history_demand)
        
        if sustained_demand_up > self.current_servers:
            # Cold Start Check: Only scale out if we have enough data points
            if not is_warming_up:
                # Check Cooldown
                if minutes_since >= cooldown_out:
                    # Step Scaling Logic
                    needed_change = sustained_demand_up - self.current_servers
                    capped_change = min(needed_change, max_step)
                    target_servers = self.current_servers + capped_change
                    
                    self.last_scale_time = timestamp_minute
                    action = "SCALE_OUT"
                else:
                    action = "WAIT_UP"
            else:
                action = "WAIT_WINDOW (Cold Start)"
        
        # Case B: Potential SCALE IN
        # We use MAX of the window to ensure sustained low demand
        elif not is_warming_up: # Only scale in if window is full
            sustained_demand_down = max(self.history_demand)
            
            # Check Hysteresis (Buffer)
            if sustained_demand_down < (self.current_servers - self.scale_in_buffer):
                # Check Cooldown
                if minutes_since >= cooldown_in:
                    # Step Scaling Logic
                    needed_change = self.current_servers - sustained_demand_down
                    capped_change = min(needed_change, max_step)
                    target_servers = self.current_servers - capped_change
                    
                    self.last_scale_time = timestamp_minute
                    action = "SCALE_IN"
                else:
                    action = "WAIT_DOWN"
            else:
                # Inside Hysteresis buffer (e.g. demand is 49, current is 50, buffer is 1)
                action = "HOLD (Hysteresis)"
        
        # Update state
        self.current_servers = int(target_servers)
        
        # 5. Cost Estimation
        real_capacity = self.current_servers * self.capacity
        dropped_req = max(0, current_load - real_capacity)
        
        unit_cost = self.cfg['cost']['server_hourly_rate']
        penalty = self.cfg['cost']['sla_penalty_per_req']
        
        cost_infra = (self.current_servers * unit_cost) / 60 
        cost_sla = dropped_req * penalty
        total_cost = cost_infra + cost_sla

        return {
            "timestamp": timestamp_minute,
            "hour": hour,
            "load": current_load,
            "forecast": forecast,
            "servers": self.current_servers,
            "raw_demand": raw_demand, 
            "capacity": real_capacity,
            "action": action,
            "is_ddos": is_ddos,
            "is_warming_up": is_warming_up,
            "dropped": dropped_req,
            "cost_infra": cost_infra,
            "cost_sla": cost_sla,
            "total_cost": total_cost,
            "safety_factor": safety_factor
        }
