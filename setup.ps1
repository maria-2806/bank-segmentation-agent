Write-Host "Starting optimized environment setup..." -ForegroundColor Cyan

# Remove old .venv if it exists to ensure a clean state with system-site-packages
if (Test-Path -Path ".venv") {
    Write-Host "Removing existing .venv directory to recreate with system-site-packages..." -ForegroundColor Yellow
    Remove-Item -Path ".venv" -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Creating Python virtual environment (.venv) inheriting system packages..." -ForegroundColor Yellow
# Using --system-site-packages to reuse system pandas, numpy, and scikit-learn
python -m venv --system-site-packages .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create virtual environment." -ForegroundColor Red
    exit 1
}

Write-Host "Activating virtual environment and upgrading pip..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to upgrade pip." -ForegroundColor Yellow
}

Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install requirements." -ForegroundColor Red
    exit 1
}

Write-Host "Python environment setup successfully completed!" -ForegroundColor Green
