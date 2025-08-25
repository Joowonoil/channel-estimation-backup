@echo off
echo Compiling ICTC 2025 paper...
cd /d D:\DNN_channel_estimation_training\papers\ICTC_2025

REM Try different possible MiKTeX locations
set MIKTEX_PATH=

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" (
    set MIKTEX_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\MiKTeX\miktex\bin\x64
)

if exist "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" (
    set MIKTEX_PATH=C:\Program Files\MiKTeX\miktex\bin\x64
)

if exist "C:\Program Files (x86)\MiKTeX\miktex\bin\pdflatex.exe" (
    set MIKTEX_PATH=C:\Program Files (x86)\MiKTeX\miktex\bin
)

if "%MIKTEX_PATH%"=="" (
    echo MiKTeX not found. Trying to use system pdflatex...
    pdflatex ICTC_2025_main.tex
    if errorlevel 1 (
        echo Failed to compile. Please install MiKTeX and restart.
        exit /b 1
    )
) else (
    echo Found MiKTeX at: %MIKTEX_PATH%
    "%MIKTEX_PATH%\pdflatex.exe" ICTC_2025_main.tex
)

echo.
echo First compilation complete.
echo Running bibtex...

if "%MIKTEX_PATH%"=="" (
    bibtex ICTC_2025_main
) else (
    "%MIKTEX_PATH%\bibtex.exe" ICTC_2025_main
)

echo.
echo Running pdflatex again for references...

if "%MIKTEX_PATH%"=="" (
    pdflatex ICTC_2025_main.tex
    pdflatex ICTC_2025_main.tex
) else (
    "%MIKTEX_PATH%\pdflatex.exe" ICTC_2025_main.tex
    "%MIKTEX_PATH%\pdflatex.exe" ICTC_2025_main.tex
)

echo.
echo Compilation complete!
echo PDF file: ICTC_2025_main.pdf