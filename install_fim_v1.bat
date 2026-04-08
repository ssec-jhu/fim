@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
::  FIM Installer for Windows
::  Double-click this file to install FIM and all dependencies.
:: ============================================================================

title FIM Installer
color 0A

echo.
echo  ============================================================
echo    FIM - Full-Field Indentation Microscopy - Installer
echo  ============================================================
echo.
echo  This script will:
echo    1. Install Git           (if not already installed)
echo    2. Install Miniconda     (if not already installed)
echo    3. Create conda env      (fim_env, Python 3.11)
echo    4. Clone the FIM repo
echo    5. Install FIM + dependencies (including PyTorch)
echo.
echo  Press Ctrl+C at any time to cancel.
echo.

:: ============================================================================
::  Step 0: Ask user for installation directory
:: ============================================================================
set "DEFAULT_DIR=%USERPROFILE%\fim"

echo  ----------------------------------------------------------
echo   Installation Directory
echo  ----------------------------------------------------------
echo.
echo  FIM will be cloned to: %DEFAULT_DIR%
echo.
set "USER_CHOICE=Y"
set /p "USER_CHOICE=  Accept this location? [Y/n]: "

if /i "!USER_CHOICE!"=="n" goto :AskCustomDir
set "INSTALL_DIR=%DEFAULT_DIR%"
goto :DirChosen

:AskCustomDir
echo.
set /p "CUSTOM_DIR=  Enter your preferred directory (e.g. D:\projects\fim): "
if "!CUSTOM_DIR!"=="" (
    echo  [!] No input provided. Using default: %DEFAULT_DIR%
    set "INSTALL_DIR=%DEFAULT_DIR%"
) else (
    set "INSTALL_DIR=!CUSTOM_DIR!"
)

:DirChosen
:: Remove trailing backslash if present
if "!INSTALL_DIR:~-1!"=="\" set "INSTALL_DIR=!INSTALL_DIR:~0,-1!"
echo.
echo  [*] Installation directory: !INSTALL_DIR!
echo.

:: ============================================================================
::  Step 1: Check / Install Git
:: ============================================================================
echo  ----------------------------------------------------------
echo   Step 1/5: Checking Git...
echo  ----------------------------------------------------------

where git >nul 2>&1
if %ERRORLEVEL% equ 0 goto :GitFound

echo  [!] Git is not installed. Installing Git for Windows...
echo.

set "GIT_INSTALLER=%TEMP%\git-installer.exe"

echo  [*] Downloading Git for Windows (64-bit)...
echo  [*] (This may take a minute...)
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/Git-2.47.1.2-64-bit.exe' -OutFile '%GIT_INSTALLER%'"

if not exist "!GIT_INSTALLER!" (
    echo  [ERROR] Failed to download Git installer.
    echo  Please install Git manually from https://git-scm.com/download/win
    echo  Then re-run this script.
    goto :ExitWithPause
)

echo  [*] Running Git installer (silent mode)...
"!GIT_INSTALLER!" /VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh"
del "!GIT_INSTALLER!" 2>nul

:: Find git.exe directly at known install locations
set "GIT_CMD="
for %%G in (
    "C:\Program Files\Git\cmd\git.exe"
    "C:\Program Files (x86)\Git\cmd\git.exe"
    "%LOCALAPPDATA%\Programs\Git\cmd\git.exe"
) do (
    if exist %%G (
        set "GIT_CMD=%%~G"
        for %%D in ("%%~dpG") do set "PATH=%%~fD;!PATH!"
    )
)

:: Also refresh PATH from registry via PowerShell (more reliable than reg query)
for /f "usebackq tokens=*" %%P in (`powershell -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"`) do set "PATH=%%P"

where git >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [OK] Git installed successfully.
    goto :GitDone
)

:: Last resort: try the found path directly
if defined GIT_CMD (
    echo  [OK] Git installed at: !GIT_CMD!
    goto :GitDone
)

echo  [WARNING] Git was installed but not detected in PATH.
echo  Please close this window, open a NEW terminal, and re-run this script.
goto :ExitWithPause

:GitFound
for /f "tokens=*" %%v in ('git --version') do echo  [OK] %%v is already installed.

:GitDone
echo.

:: ============================================================================
::  Step 2: Check / Install Miniconda
:: ============================================================================
echo  ----------------------------------------------------------
echo   Step 2/5: Checking Miniconda / Anaconda...
echo  ----------------------------------------------------------

:: Method 1: Check PATH
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 goto :CondaFound

:: Method 2: Check common install locations
set "CONDA_SEARCH_DIRS=miniconda3 Miniconda3 anaconda3 Anaconda3"
for %%N in (%CONDA_SEARCH_DIRS%) do (
    if exist "%USERPROFILE%\%%N\Scripts\conda.exe" (
        echo  [*] Found conda at: %USERPROFILE%\%%N
        set "CONDA_ROOT=%USERPROFILE%\%%N"
        goto :CondaActivateExisting
    )
)
for %%N in (%CONDA_SEARCH_DIRS%) do (
    if exist "C:\%%N\Scripts\conda.exe" (
        echo  [*] Found conda at: C:\%%N
        set "CONDA_ROOT=C:\%%N"
        goto :CondaActivateExisting
    )
)
for %%N in (%CONDA_SEARCH_DIRS%) do (
    if exist "C:\ProgramData\%%N\Scripts\conda.exe" (
        echo  [*] Found conda at: C:\ProgramData\%%N
        set "CONDA_ROOT=C:\ProgramData\%%N"
        goto :CondaActivateExisting
    )
)

:: Not found anywhere - install it
goto :InstallConda

:CondaActivateExisting
set "PATH=!CONDA_ROOT!\Scripts;!CONDA_ROOT!\condabin;!CONDA_ROOT!;!PATH!"
call "!CONDA_ROOT!\Scripts\activate.bat" "!CONDA_ROOT!" 2>nul
where conda >nul 2>&1
if %ERRORLEVEL% equ 0 goto :CondaFound
echo  [WARNING] Found conda directory but could not activate it.
goto :InstallConda

:InstallConda
echo  [!] Miniconda is not installed. Installing Miniconda...
echo.

set "CONDA_INSTALLER=%TEMP%\miniconda-installer.exe"
set "CONDA_INSTALL_DIR=%USERPROFILE%\miniconda3"

echo  [*] Downloading Miniconda (64-bit)...
echo  [*] (This may take a minute...)
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%CONDA_INSTALLER%'"

if not exist "!CONDA_INSTALLER!" (
    echo  [ERROR] Failed to download Miniconda installer.
    echo  Please install manually from https://docs.conda.io/en/latest/miniconda.html
    echo  Then re-run this script.
    goto :ExitWithPause
)

echo  [*] Installing Miniconda to: !CONDA_INSTALL_DIR!
echo      (This may take a few minutes...)
start /wait "" "!CONDA_INSTALLER!" /S /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /D=!CONDA_INSTALL_DIR!
del "!CONDA_INSTALLER!" 2>nul

:: Activate the fresh install
set "PATH=!CONDA_INSTALL_DIR!\Scripts;!CONDA_INSTALL_DIR!\condabin;!CONDA_INSTALL_DIR!;!PATH!"
call "!CONDA_INSTALL_DIR!\Scripts\activate.bat" "!CONDA_INSTALL_DIR!" 2>nul

where conda >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [OK] Miniconda installed successfully.
    goto :CondaDone
)

echo  [ERROR] Miniconda installed but conda not found in PATH.
echo  Please close this window, open a NEW terminal, and re-run this script.
goto :ExitWithPause

:CondaFound
for /f "tokens=*" %%v in ('conda --version 2^>nul') do echo  [OK] %%v is already installed.

:CondaDone
echo.

:: ============================================================================
::  Step 3: Create conda environment
:: ============================================================================
echo  ----------------------------------------------------------
echo   Step 3/5: Setting up conda environment (fim_env)...
echo  ----------------------------------------------------------

:: Initialize conda shell hooks for activate to work
call conda init cmd.exe >nul 2>&1

:: Check if fim_env already exists
call conda info --envs 2>nul | findstr /C:"fim_env" >nul 2>&1
if %ERRORLEVEL% neq 0 goto :CreateEnv

echo  [*] Conda environment 'fim_env' already exists.
set "RECREATE=N"
set /p "RECREATE=  Recreate it from scratch? [y/N]: "
if /i "!RECREATE!"=="y" (
    echo  [*] Removing existing fim_env...
    call conda deactivate 2>nul
    call conda env remove -n fim_env -y
    goto :CreateEnv
)

echo  [OK] Using existing fim_env.
goto :ActivateEnv

:CreateEnv
echo  [*] Accepting conda channel Terms of Service...
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>&1
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>&1
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>&1
echo  [OK] Terms accepted.
echo.
echo  [*] Creating conda environment: fim_env (Python 3.11)...
call conda create -n fim_env python=3.11 -y
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Failed to create conda environment.
    goto :ExitWithPause
)
echo  [OK] Environment created.

:ActivateEnv
echo.
echo  [*] Activating fim_env...
call conda activate fim_env

:: Fallback activation if the above didn't work
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [*] Trying alternative activation...
    for /f "tokens=*" %%p in ('conda info --base 2^>nul') do set "CONDA_BASE=%%p"
    if defined CONDA_BASE call "!CONDA_BASE!\Scripts\activate.bat" fim_env
)

python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Python not available after activating fim_env.
    goto :ExitWithPause
)
for /f "tokens=*" %%v in ('python --version') do echo  [OK] Active Python: %%v
echo.

:: ============================================================================
::  Step 4: Clone (or update) the FIM repository
:: ============================================================================
echo  ----------------------------------------------------------
echo   Step 4/5: Getting FIM source code...
echo  ----------------------------------------------------------

:: Create parent directory if needed
for %%D in ("!INSTALL_DIR!") do set "PARENT_DIR=%%~dpD"
if not exist "!PARENT_DIR!" mkdir "!PARENT_DIR!" 2>nul

:: Check if repo already exists
if exist "!INSTALL_DIR!\.git" goto :RepoExists

:: Check if dir exists but isn't a repo
if exist "!INSTALL_DIR!" goto :DirExistsNotRepo

:: Fresh clone
echo  [*] Cloning FIM repository...
git clone https://github.com/ssec-jhu/fim.git "!INSTALL_DIR!"
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Failed to clone repository. Check your internet connection.
    goto :ExitWithPause
)
echo  [OK] Repository cloned.
goto :RepoDone

:RepoExists
echo  [*] FIM repo already exists at: !INSTALL_DIR!
set "PULL_UPDATE=Y"
set /p "PULL_UPDATE=  Pull latest changes? [Y/n]: "
if /i "!PULL_UPDATE!"=="n" goto :RepoDone
echo  [*] Pulling latest changes...
pushd "!INSTALL_DIR!"
git pull
popd
echo  [OK] Repository updated.
goto :RepoDone

:DirExistsNotRepo
echo  [WARNING] Directory !INSTALL_DIR! exists but is not a git repo.
set "INSTALL_DIR=!INSTALL_DIR!\fim"
echo  [*] Cloning into !INSTALL_DIR! instead...
git clone https://github.com/ssec-jhu/fim.git "!INSTALL_DIR!"
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Failed to clone repository. Check your internet connection.
    goto :ExitWithPause
)

:RepoDone
echo.

:: ============================================================================
::  Step 5: Install FIM and all dependencies
:: ============================================================================
echo  ----------------------------------------------------------
echo   Step 5/5: Installing FIM and dependencies...
echo  ----------------------------------------------------------
echo  (This may take several minutes - PyTorch is a large download)
echo.

pushd "!INSTALL_DIR!"
echo  [*] Upgrading pip...
call python -m pip install --upgrade pip
echo.
call pip install -e .
if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] pip install failed. Check the error messages above.
    popd
    goto :ExitWithPause
)
popd

echo.
echo  [OK] FIM installed successfully!
echo.

:: ============================================================================
::  Verify installation
:: ============================================================================
echo  ----------------------------------------------------------
echo   Verifying installation...
echo  ----------------------------------------------------------

python -c "import fim; print('  [OK] fim package importable')" 2>nul
if %ERRORLEVEL% neq 0 echo  [WARNING] Could not import fim. Installation may be incomplete.

python -c "import torch; print('  [OK] PyTorch', torch.__version__, 'installed')" 2>nul
if %ERRORLEVEL% neq 0 echo  [WARNING] PyTorch not found. Some features may not work.

where fim-ui >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  [OK] fim-ui command available.
) else (
    echo  [WARNING] fim-ui command not found in PATH.
)

echo.

:: ============================================================================
::  Done - ask about launching
:: ============================================================================
echo  ============================================================
echo    Installation Complete!
echo  ============================================================
echo.
echo  FIM is installed at: !INSTALL_DIR!
echo.
echo  There are two ways to run FIM:
echo.
echo  --- Option 1: Web UI (recommended for most users) ---
echo    1. Open a terminal (cmd or Anaconda Prompt)
echo    2. conda activate fim_env
echo    3. fim-ui
echo    4. Open http://127.0.0.1:8000 in your browser
echo.
echo  --- Option 2: CLI (for advanced/scripting use) ---
echo    1. Open a terminal (cmd or Anaconda Prompt)
echo    2. conda activate fim_env
echo    3. cd !INSTALL_DIR!
echo    4. python -m fim.app.cli list-steps
echo    5. python -m fim.app.cli run tracking --set out_dir=C:\output
echo.
set "LAUNCH_NOW=Y"
set /p "LAUNCH_NOW=  Launch FIM web UI now? [Y/n]: "
if /i "!LAUNCH_NOW!"=="n" goto :SkipLaunch

echo.
echo  [*] Starting FIM web UI...
echo  [*] Opening browser to http://127.0.0.1:8000 in 5 seconds...
echo  [*] Press Ctrl+C to stop the server.
echo.

:: Open browser after a short delay
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://127.0.0.1:8000"

:: Hand off to a new interactive cmd session.
:: This exits the batch context so Ctrl+C won't ask "Terminate batch job?"
cmd /k "cd /d !INSTALL_DIR! & call conda activate fim_env & echo. & fim-ui & echo. & echo  ---------------------------------------------------------- & echo   FIM UI stopped. You are in: !INSTALL_DIR! & echo   To restart: fim-ui & echo   To exit:    type exit & echo  ----------------------------------------------------------"
goto :EOF

:SkipLaunch
echo.
echo  [*] You can launch FIM anytime by running:
echo      conda activate fim_env ^& fim-ui
echo.

:ExitWithPause
echo.
echo  Press any key to exit...
pause >nul
exit /b 0

:: end of script
