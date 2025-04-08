#!/bin/bash

source .venv/bin/activate
current_date_time=$(date +"%Y-%m-%d %H:%M:%S")
nohup python3 src/train.py > "logs/$current_date_time.out" &