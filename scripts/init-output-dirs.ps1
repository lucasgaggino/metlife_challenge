# Crea carpetas de salida para el pipeline (Windows / Docker Desktop).
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
@(
    "models", "results", "logs",
    "models\prod", "results\prod", "logs\prod"
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}
Write-Host "OK: models/, results/, logs/ creados. Ejecuta: docker compose up --build"
