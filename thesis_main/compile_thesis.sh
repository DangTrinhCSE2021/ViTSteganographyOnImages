#!/bin/bash
# Thesis Compilation Script
# This script compiles the complete thesis document

echo "========================================"
echo "  Enhanced ViT Steganography Thesis"
echo "      Compilation Script"
echo "========================================"

echo ""
echo "Step 1: Generating figures and diagrams..."
python generate_figures.py

echo ""
echo "Step 2: Compiling LaTeX document..."
echo "Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "Running bibtex for bibliography..."
bibtex main

echo ""
echo "Running pdflatex (second pass)..."
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "Running pdflatex (final pass)..."
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "========================================"
echo "  Compilation Complete!"
echo "========================================"
echo ""
echo "Output: main.pdf"
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot

echo ""
echo "Done! Your thesis is ready in main.pdf"
