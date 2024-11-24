#!/bin/bash

current_date_time=$(date +"%Y-%m-%d %H:%M:%S")
nohup python3 main.py > "logs/$current_date_time.out" &