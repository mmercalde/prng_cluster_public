#!/bin/bash
python3 coordinator.py \
    test_survivors_100.json \
    train_history.json \
    holdout_history.json \
    scorer_jobs.json \
    192.168.3.120 192.168.3.154
