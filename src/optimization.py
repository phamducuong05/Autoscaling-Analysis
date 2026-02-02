import math
import pandas as pd
from src.utils import load_config

class Autoscaler:
    """
    Core Inference Engine for Autoscaling Decisions.
    Implements:
    1. Dynamic Safety Factors based on Time of Day (Volatility Analysis)
    2. Hybrid Scaling (Max of Predictive & Reactive)
    3. Cooldown/Hysteresis for Stability
    4. DDoS Anomaly Detection
    """
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
        """
        Simple anomaly detection: If current load exceeds forecast by a huge multiplier.
        """
        multiplier = self.cfg['anomaly']['ddos_multiplier']
        # Avoid false positives on very low traffic (e.g. forecast=2, load=12)
        if forecast > 10 and current_load > forecast * multiplier:
            return True
        return False

    def decide_scale(self, timestamp_minute, current_load, forecast, hour):
        """
        Main decision loop for a single time step.
        Returns a dictionary with decision details.
        """
        # 1. Anomaly Detection
        is_ddos = self.detect_ddos(current_load, forecast)
        
        safety_factor = self.get_safety_factor(hour)
        
        # 2. Logic Calculation
        if is_ddos:
            # During DDoS, do not apply safety factor (avoid amplifying the attack)
            # Cap the max servers to prevent wallet draining
            raw_demand = math.ceil(current_load / self.capacity)
            raw_demand = min(raw_demand, self.cfg['anomaly']['ddos_max_servers'])
            safety_factor = 1.0 
        else:
            # Hybrid Scaling: Plan for the worst of (Forecast vs Actual)
            demand_pred = math.ceil((forecast * safety_factor) / self.capacity)
            demand_react = math.ceil((current_load * safety_factor) / self.capacity)
            raw_demand = max(demand_pred, demand_react)
            
            # Circuit Breaker
            raw_demand = min(raw_demand, self.max_servers)

        # Ensure min servers
        raw_demand = max(raw_demand, self.min_servers)

        # 3. Stability & Cooldown Logic
        minutes_since = timestamp_minute - self.last_scale_time
        action = "HOLD"
        cooldown_out = self.cfg['cooldown']['scale_out_minutes']
        cooldown_in = self.cfg['cooldown']['scale_in_minutes']
        
        target_servers = self.current_servers

        # Scale OUT (Increase)
        if raw_demand > self.current_servers:
            if minutes_since >= cooldown_out:
                target_servers = raw_demand
                self.last_scale_time = timestamp_minute
                action = "SCALE_OUT"
            else:
                action = "WAIT_UP"
        
        # Scale IN (Decrease)
        elif raw_demand < self.current_servers:
            if minutes_since >= cooldown_in:
                target_servers = raw_demand
                self.last_scale_time = timestamp_minute
                action = "SCALE_IN"
            else:
                action = "WAIT_DOWN"
        
        # Update state
        self.current_servers = int(target_servers)
        
        # 4. Cost Estimation (Observation)
        real_capacity = self.current_servers * self.capacity
        dropped_req = max(0, current_load - real_capacity)
        
        unit_cost = self.cfg['cost']['server_hourly_rate']
        penalty = self.cfg['cost']['sla_penalty_per_req']
        
        cost_infra = (self.current_servers * unit_cost) / 60 # per minute
        cost_sla = dropped_req * penalty
        total_cost = cost_infra + cost_sla

        return {
            "timestamp": timestamp_minute,
            "hour": hour,
            "load": current_load,
            "forecast": forecast,
            "servers": self.current_servers,
            "capacity": real_capacity,
            "action": action,
            "is_ddos": is_ddos,
            "dropped": dropped_req,
            "cost_infra": cost_infra,
            "cost_sla": cost_sla,
            "total_cost": total_cost,
            "safety_factor": safety_factor
        }
