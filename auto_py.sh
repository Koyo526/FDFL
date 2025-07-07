#!/bin/bash

ARRAY=(1 2 3)
for i in "${ARRAY[@]}"; do
    echo "Running Random_DataSize_FL.py with $i"
    python3 Random_DataSize_FL.py
done