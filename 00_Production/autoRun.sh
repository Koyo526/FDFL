#!/bin/bash

ARRAY=(1 2 3 4 5 6 7 8 9 10)
for i in "${ARRAY[@]}"; do
    echo "Running base.py with $i"
    python3 base.py
    echo "Running randomFreeRider.py with $i"
    python3 randomFreeRider.py
    echo "Running deltaFreeRider with $i"
    python3 deltaFreeRider.py
    echo "Running changeDataSize.py with $i"
    python3 changeDataSize.py
done
