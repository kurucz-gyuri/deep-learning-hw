#!/bin/sh
FILE=deep_learners.tex
pdflatex "$FILE"
bibtex "$FILE"
pdflatex "$FILE"
pdflatex "$FILE"
