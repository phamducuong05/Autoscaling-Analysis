import requests
import json
import logging

# Config
API_URL = "http://127.0.0.1:8000"

def test_api_contract():
    print("Testing API Contract...")
    
    # 1. Test Forecast
    print("\n[1] Testing /forecast...")
    forecast_payload = {
        "timestamp_minute": 100,
        "recent_history": [50.0] * 12,
        "error_history": [0.0] * 12,
        "hour_sin_history": [0.5] * 12,
        "hour_cos_history": [0.5] * 12,
        "is_weekend_history": [0.0] * 12,
        "actual_load_current": 55.0
    }
    
    try:
        resp = requests.post(f"{API_URL}/forecast", json=forecast_payload)
        resp.raise_for_status()
        data = resp.json()
        print(f"✅ /forecast Response: {json.dumps(data, indent=2)}")
        
        # Verify fields
        assert 'forecast_load' in data, "Missing forecast_load"
        assert 'sigma' in data, "Missing sigma"
        assert 'cv' in data, "Missing cv"
        
        forecast_val = data['forecast_load']
        sigma = data['sigma']
        cv = data['cv']
        
    except Exception as e:
        print(f"❌ /forecast Failed: {e}")
        return

    # 2. Test Scaling
    print("\n[2] Testing /recommend-scaling...")
    scaling_payload = {
        "timestamp_minute": 100,
        "current_load": 55.0,
        "forecast_load": forecast_val,
        "hour_of_day": 10,
        "sigma": sigma,
        "cv": cv
    }
    
    try:
        resp = requests.post(f"{API_URL}/recommend-scaling", json=scaling_payload)
        resp.raise_for_status()
        data = resp.json()
        print(f"✅ /recommend-scaling Response: {json.dumps(data, indent=2)}")
        
        # Verify Critical Dashboard Fields
        required_fields = ['timestamp', 'servers', 'action', 'dropped', 'capacity']
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            print(f"❌ Missing Fields: {missing}")
        else:
            print("✅ All Dashboard Contract Fields Present!")
            
    except Exception as e:
        print(f"❌ /recommend-scaling Failed: {e}")

if __name__ == "__main__":
    test_api_contract()
