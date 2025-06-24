@echo off
echo === LaTeX Compilation Script ===
cd /d "%~dp0"

echo Step 1: First LaTeX run...
pdflatex -interaction=nonstopmode main.tex

echo Step 2: Processing bibliography with Biber...
biber main

echo Step 3: Second LaTeX run...
pdflatex -interaction=nonstopmode main.tex

echo Step 4: Final LaTeX run...
pdflatex -interaction=nonstopmode main.tex

echo.
echo === Compilation Complete ===
echo Output: main.pdf
pause
