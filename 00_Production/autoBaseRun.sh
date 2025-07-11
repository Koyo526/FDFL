#!/bin/bash

ARRAY=(1 2 3 4 5 6 7 8 9 10)
for i in "${ARRAY[@]}"; do
    echo "Running base.py with $i"
    python base.py $i
done
