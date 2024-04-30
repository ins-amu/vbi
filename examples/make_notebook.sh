#!/bin/bash

# This script does the following:
# 1. loops over all .py files in the current directory and convert them to .ipynb using p2j
# 2. move them to the notebooks directory
# 3. finally run them and save the results into the same directory

# -----------------------------------------------------------------------------
# prerequisites:
# install p2j: pip install p2j
# install jupyter: pip install jupyter
# -----------------------------------------------------------------------------

directory="notebooks"

if [ ! -d "$directory" ]; then
    mkdir "$directory"
    echo "Directory '$directory' created."
fi

cp helpers.py notebooks/helpers.py

for f in *.py
do

    if [[ "$f" == "helpers.py" ]]; then
        continue
    fi

    # convert to .ipynb
    p2j -o $f
    f="${f%.*}"
    mv $f.ipynb notebooks/$f.ipynb
    # execute notebook and save results
    cd notebooks
    jupyter nbconvert --to notebook --execute $f.ipynb --output=$f.ipynb --ExecutePreprocessor.timeout=-1
    cd ..

done
