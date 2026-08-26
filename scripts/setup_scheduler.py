#!/usr/bin/env python3
"""
Set up Windows Task Scheduler for hourly real data collection.
"""
import sys
import os
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
batch_file = project_root / "scripts" / "schedule_collection.bat"

print("=" * 80)
print("WINDOWS TASK SCHEDULER SETUP")
print("=" * 80)
print()
print("To set up hourly collection, run the following command as Administrator:")
print()
print(f'  schtasks /create /tn "AQI_Predictor_Collection" /tr "{batch_file}" /sc hourly /st 00:00 /f')
print()
print("Or use Task Scheduler GUI:")
print("  1. Open Task Scheduler (taskschd.msc)")
print("  2. Create Basic Task")
print("  3. Name: AQI_Predictor_Collection")
print("  4. Trigger: Daily, repeat every 1 hour")
print("  5. Action: Start a program")
print("  6. Program: C:\\MiniConda\\condabin\\conda.bat")
print(f'  7. Arguments: run -n aqi-predictor python "{project_root / "scripts" / "collect_real_data.py"}"')
print("  8. Start in: " + str(project_root))
print()
print("To verify the task is scheduled:")
print('  schtasks /query /tn "AQI_Predictor_Collection"')
print()
print("To remove the task:")
print('  schtasks /delete /tn "AQI_Predictor_Collection" /f')
print()
print("=" * 80)
