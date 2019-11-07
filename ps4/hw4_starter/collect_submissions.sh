#!/bin/bash
rm -f hw4.zip
zip -R hw4.zip \
    "dynamic_programming/*.ipynb" \
    "dynamic_programming/*.py" \
    "dynamic_programming/*.pdf" \
    "q_learning/*.pdf"\
    "q_learning/*.py"\
    "q_learning/*/*.py"\
    "q_learning/*.ipynb"
echo "Submission file hw4.zip created."
