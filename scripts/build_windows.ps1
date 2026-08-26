$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

uv sync --extra dev --frozen
uv run pytest -q
uv run pyinstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "GPRLayerAudit" `
    --exclude-module pyqtgraph.opengl `
    --exclude-module pyqtgraph.examples `
    --exclude-module matplotlib.tests `
    --hidden-import scipy.signal `
    --hidden-import scipy.ndimage `
    launcher.py

uv run pyinstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "GPRLayerAudit-Portable" `
    --exclude-module pyqtgraph.opengl `
    --exclude-module pyqtgraph.examples `
    --exclude-module matplotlib.tests `
    --hidden-import scipy.signal `
    --hidden-import scipy.ndimage `
    launcher.py

Write-Host "Application built at dist\GPRLayerAudit\GPRLayerAudit.exe"
Write-Host "Portable application built at dist\GPRLayerAudit-Portable.exe"
Write-Host "Compile installer\GPRLayerAudit.iss with Inno Setup for the signed-ready installer."
