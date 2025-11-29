#!/bin/bash
cd ~/distributed_prng_analysis || exit
git add .
git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
git push
