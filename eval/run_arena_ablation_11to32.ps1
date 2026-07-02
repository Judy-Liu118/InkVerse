# eval/run_arena_ablation_11to32.ps1
# Refill arena ablation for themes 11-32 (22 themes),
# backend = qwen-image-2.0-pro-2026-04-22
#
# Usage (from project root D:\PythonProject\InkVerse):
#   .\eval\run_arena_ablation_11to32.ps1

$ErrorActionPreference = "Continue"

# Force console to UTF-8 so Chinese in python stdout is not garbled.
# Only affects this PowerShell session; system code page unchanged.
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$Python = "D:\anaconda3\envs\poetry_env\python.exe"
if (-not (Test-Path -Path $Python)) {
    Write-Host "[!] Python not found: $Python"
    exit 1
}

$LogDir = "outputs\eval"
if (-not (Test-Path -Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Backend = "bailian:qwen-image-2.0-pro-2026-04-22"
$LogA = Join-Path $LogDir "print_arm_a_ablation_11to32_qwen-image-2.0-pro-2026-04-22"
$LogB = Join-Path $LogDir "print_arm_b_ablation_11to32_qwen-image-2.0-pro-2026-04-22"

Write-Host "==================== arm A (max_poem_rounds=0, no arena) ===================="
& $Python -m eval.sweep_pairwise_win_delta --n 32 --offset 10 --deltas 0.17 --max-poem-rounds 0 --image-backend $Backend | Tee-Object -FilePath $LogA
$exitA = $LASTEXITCODE
Write-Host "[arm A exit code] $exitA"

if ($exitA -ne 0) {
    Write-Host "[!] arm A failed (exit=$exitA). Log: $LogA"
    Write-Host "[!] Skipping arm B to save quota."
    exit 1
}

Write-Host ""
Write-Host "==================== arm B (max_poem_rounds=2, with arena) ===================="
& $Python -m eval.sweep_pairwise_win_delta --n 32 --offset 10 --deltas 0.17 --max-poem-rounds 2 --image-backend $Backend | Tee-Object -FilePath $LogB
$exitB = $LASTEXITCODE
Write-Host "[arm B exit code] $exitB"

if ($exitB -ne 0) {
    Write-Host "[!] arm B failed (exit=$exitB). Log: $LogB"
    exit 1
}

Write-Host ""
Write-Host "==================== ALL DONE ===================="
Write-Host "arm A log: $LogA"
Write-Host "arm B log: $LogB"
Write-Host ""
Write-Host "Next: tell Claude 'themes 11-32 done', it will build n=32 agg and update the report."