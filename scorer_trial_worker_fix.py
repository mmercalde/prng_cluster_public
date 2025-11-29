# After line 168 (after y_train = engine.scorer.batch_score(...))
# Add this line to extract just the scores:

# ORIGINAL (line ~168):
y_train = engine.scorer.batch_score(
    seeds=seeds_to_score,
    lottery_history=train_history,
    use_dual_gpu=False
)

# ADD THIS LINE RIGHT AFTER:
y_train = [item['score'] if isinstance(item, dict) else item for item in y_train]
