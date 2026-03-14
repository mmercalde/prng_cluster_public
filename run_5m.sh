source ~/venvs/torch/bin/activate
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline   --start-step 0 --end-step 1   --params '{lottery_file:daily3.json,window_trials:200,resume_study:true,study_name:window_opt_1773408949,max_seeds:5000000,enable_pruning:true,n_parallel:2,use_persistent_workers:true,worker_pool_size:4,seed_cap_nvidia:5000000,seed_cap_amd:2000000}'   > ~/distributed_prng_analysis/logs/step01_200trial_5M.log 2>&1
