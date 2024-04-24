@echo off
setlocal enabledelayedexpansion

set PACKAGE_LIST=requirements.txt
set FAIL_LOG=failed_installations.txt

if not exist %PACKAGE_LIST% (
    echo Error: Package list file "%PACKAGE_LIST%" not found.
    exit /b 1
)

if exist %FAIL_LOG% (
    del %FAIL_LOG%
)

for /f %%i in (%PACKAGE_LIST%) do (
    echo Installing package: %%i
    pip install %%i

    if !errorlevel! neq 0 (
        echo Failed to install: %%i >> %FAIL_LOG%
    )
)

echo Installation complete.

endlocal
