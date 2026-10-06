@echo off
setlocal
if "%~1"=="" goto usage
if not exist "%~1" (
  echo ComfyUI Python was not found: "%~1"
  exit /b 2
)
set "PDMD_CONFIG=%~2"
if not defined PDMD_CONFIG set "PDMD_CONFIG=%~dp0config.local.json"
"%~1" -B -X utf8 "%~dp0scripts\setup.py" --config "%PDMD_CONFIG%" %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:usage
echo Usage: setup.bat "ComfyUI Python executable" "config.local.json" [--check]
echo Copy config.example.json to config.local.json and set your existing model paths.
echo Without --check, setup installs dependencies and X2 nodes into the selected ComfyUI,
echo prepares sample images, and reuses an existing fused model or creates one on CPU.
exit /b 2
