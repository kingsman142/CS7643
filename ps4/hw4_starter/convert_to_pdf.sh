#!/bin/bash
rm -rf dynamic_programming/dp.html
rm -rf q_learning/q_learning.html

jupyter nbconvert --to html dynamic_programming/dp.ipynb
jupyter nbconvert --to html q_learning/q_learning.ipynb

if [ "$1" == "--no-pdf" ]; then
    echo ""
    echo "WARNING: HTML generated without PDF conversion."
    echo "Please open the HTML files in a browser and save them as PDFs for submission."
    exit 1
fi

if ! [ -x "$(command -v wkhtmltopdf)" ]; then
    echo 'Error: wkhtmltopdf command not found. Either install from https://wkhtmltopdf.org/downloads.html or manually convert HTML files to PDF from browser.' >&2
    exit 1
fi

wkhtmltopdf dynamic_programming/dp.html dynamic_programming/dp.pdf
wkhtmltopdf q_learning/q_learning.html q_learning/q_learning.pdf
