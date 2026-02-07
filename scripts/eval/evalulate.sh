uv run scripts/historyvla/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/new_ckpts/historypi05_bench_obs16/bg0-obs16-symbolic-simple-subgoal/79999 --policy.config=historypi05_bench_obs16



python examples/history_bench_sim/eval.py --args.port=8001 --args.model_seed=7 --args.model_ckpt_id=79999 \
    --args.policy_name=<your_policy_name> \
    --args.use_qwenvl --args.subgoal_type=grounded_subgoal
