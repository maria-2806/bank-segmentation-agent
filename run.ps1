Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Starting Customer Segmentation & Personalization Agent" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Check virtual environment
if (-not (Test-Path -Path ".venv\Scripts\python.exe")) {
    Write-Host "Virtual environment (.venv) not found. Please run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# 2. Check synthetic dataset
if (-not (Test-Path -Path "banking_customers.csv")) {
    Write-Host "Synthetic customer dataset not found. Generating banking_customers.csv..." -ForegroundColor Yellow
    & .venv\Scripts\python.exe data_generator.py
}

# 3. Check Ollama local status
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -ErrorAction Stop
    Write-Host "[OK] Local Ollama service is online!" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Could not connect to local Ollama on http://localhost:11434" -ForegroundColor Yellow
    Write-Host "Please ensure Ollama is installed and running (run 'ollama run llama3.2' in terminal)." -ForegroundColor Yellow
}

Write-Host "Launching FastAPI backend and serving React UI at http://localhost:8000 ..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Gray

& .venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
