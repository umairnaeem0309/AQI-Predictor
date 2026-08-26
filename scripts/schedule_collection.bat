@echo off
REM AQI Predictor - Hourly Real Data Collection
REM Scheduled via Windows Task Scheduler
REM Runs every hour using Python 3.11 conda environment

REM Activate conda environment
call C:\MiniConda\condabin\conda.bat activate aqi-predictor

REM Run collection
python "%~dp0collect_real_data.py"

REM Exit with collection status
exit /b %errorlevel%
