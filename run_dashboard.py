"""
Dashboard Launcher for Employee Performance Predictor

This script launches the Streamlit dashboard for the Employee Performance Predictor.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard."""
    print("Starting Employee Performance Predictor Dashboard...")
    print("Please wait for the dashboard to open in your browser...")
    
    # Change to the outputs directory
    os.chdir('outputs')
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        print("Make sure you have installed the requirements:")
        print("pip install -r requirements_dashboard.txt")
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")

if __name__ == "__main__":
    main()
