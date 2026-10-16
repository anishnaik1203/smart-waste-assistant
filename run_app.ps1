# Smart Waste Assistant - PowerShell Launcher
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "      Smart Waste Classification & Recycling Assistant" -ForegroundColor Green
Write-Host "         AI & Data Science Engineering Capstone" -ForegroundColor Yellow
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
$venvStreamlit = Join-Path $scriptDir ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $venvStreamlit)) {
    Write-Host "[ERROR] Virtual environment streamlit not found at $venvStreamlit" -ForegroundColor Red
    Write-Host "Please create the venv and install dependencies from requirements.txt."
    Exit 1
}

Write-Host "Launching Streamlit application on http://localhost:8501 ..." -ForegroundColor Green
& $venvStreamlit run (Join-Path $scriptDir "app.py")
