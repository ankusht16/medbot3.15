@echo off

REM Read each line from the text file
for /f "delims=" %%i in (requirements.txt) do (
    REM Install the package using pip
    pip3 install %%i
)