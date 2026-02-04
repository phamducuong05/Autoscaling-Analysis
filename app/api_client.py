"""
API Client Module
Extracted from dashboard.py - API call logic
"""
import requests
import streamlit as st

class AutoscalingAPIClient:
    """Client for Autoscaling API endpoints"""
    
    def __init__(self, base_url):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of API (e.g., http://localhost:8000)
        """
        self.base_url = base_url
    
    def get_forecast(self, payload):
        """
        Call /forecast endpoint
        
        Args:
            payload: dict with forecast request data
            
        Returns:
            dict with forecast response or None if error
        """
        try:
            resp = requests.post(f"{self.base_url}/forecast", json=payload)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.warning(f"Forecast API Error: {resp.status_code}")
                return None
        except Exception as e:
            st.error(f"API Connection Error: {e}")
            return None
    
    def get_scaling_decision(self, payload):
        """
        Call /recommend-scaling endpoint
        
        This is EXACT copy from dashboard.py call_api() function
        
        Args:
            payload: dict with scaling request data
            
        Returns:
            dict with scaling decision or None if error
        """
        try:
            resp = requests.post(f"{self.base_url}/recommend-scaling", json=payload)
            if resp.status_code == 200:
                return resp.json()
            else:
                st.error(f"API Error: {resp.status_code}")
                return None
        except Exception as e:
            st.error(f"Connection Error: Is the API running? {e}")
            return None
