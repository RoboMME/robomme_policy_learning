# Manual evaluation (per model)

## Outline

- [π₀.₅ baseline](#π₀₅-baseline)
- [MemER](#memer)
- [Symbolic MME-VLA](#symbolic-mme-vla)
  - [SimpleSG + Oracle](#simplesg--oracle)
  - [SimpleSG + QwenVL](#simplesg--qwenvl)
  - [SimpleSG + Gemini](#simplesg--gemini)
  - [GroundSG + Oracle](#groundsg--oracle)
  - [GroundSG + QwenVL](#groundsg--qwenvl)
  - [GroundSG + Gemini](#groundsg--gemini)
- [Perceptual MME-VLA](#perceptual-mme-vla)
  - [TokenDrop + Context](#tokendrop--context)
  - [TokenDrop + Modulation](#tokendrop--modulation)
  - [TokenDrop + Expert](#tokendrop--expert)
  - [FrameSamp + Context](#framesamp--context)
  - [FrameSamp + Modulation](#framesamp--modulation)
  - [FrameSamp + Expert](#framesamp--expert)
- [Recurrent MME-VLA](#recurrent-mme-vla)
  - [TTT + Context](#ttt--context)
  - [TTT + Modulation](#ttt--modulation)
  - [TTT + Expert](#ttt--expert)
  - [RMT + Context](#rmt--context)
  - [RMT + Modulation](#rmt--modulation)
  - [RMT + Expert](#rmt--expert)
- [Other Hints](#other-hints)


## π₀.₅ baseline
```
# terminal 0
uv run scripts/serve_policy.py --seed=7 --port=8001 policy:checkpoint --policy.dir=runs/ckpts/pi05_baseline/pi05_baseline/79999 --policy.config=pi05_baseline

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=pi05_baseline --args.model_ckpt_id=79999 --args.no-use-history
```

You can change the `--policy.dir` to load different ckpts, change the  `seed` and `ckpt_id` to evaluate on different checkpoints and seeds, then gather results with `scripts/compute_results.py`.


## MemER
MemER can be viewed as a combined use of symbolic and perceptual memory.

```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-grounded-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=grounded_subgoal --args.use-memer 
```


## Symbolic MME-VLA

### SimpleSG + Oracle
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-simple-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-simple-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=simple_subgoal --args.use-oracle 
```

### SimpleSG + QwenVL
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-simple-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-simple-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=simple_subgoal --args.use-qwenvl 
### SimpleSG + Gemini
Set the `GOOGLE_API_KEY` environment variable when using Gemini.
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-simple-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-simple-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=simple_subgoal --args.use-gemini 
```

### GroundSG + Oracle
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-grounded-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=grounded_subgoal --args.use-oracle 
```

### GroundSG + QwenVL
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-grounded-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=grounded_subgoal --args.use-qwenvl 
### GroundSG + Gemini
Set the `GOOGLE_API_KEY` environment variable when using Gemini.
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=symbolic-grounded-subgoal --args.model_ckpt_id=79999  --args.subgoal-type=grounded_subgoal --args.use-gemini 
```

## Perceptual MME-VLA

### TokenDrop + Context
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-tokendrop-context/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-tokendrop-context --args.model_ckpt_id=79999
```

### TokenDrop + Modulation
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-tokendrop-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-tokendrop-modul --args.model_ckpt_id=79999
```

### TokenDrop + Expert
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-tokendrop-expert/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-tokendrop-expert --args.model_ckpt_id=79999
```

### FrameSamp + Context
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-framesamp-context/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-framesamp-context --args.model_ckpt_id=79999
```

### FrameSamp + Modulation
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-framesamp-modul --args.model_ckpt_id=79999
```

### FrameSamp + Expert
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/perceptual-framesamp-expert/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=perceptual-framesamp-expert --args.model_ckpt_id=79999
```


## Recurrent MME-VLA

### TTT + Context
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-ttt-context/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-ttt-context --args.model_ckpt_id=79999
```

### TTT + Modulation
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-ttt-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-ttt-modul --args.model_ckpt_id=79999
```

### TTT + Expert
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-ttt-expert/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-ttt-expert --args.model_ckpt_id=79999
```

### RMT + Context
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-rmt-context/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-rmt-context --args.model_ckpt_id=79999
```

### RMT + Modulation
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-rmt-modul/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-rmt-modul --args.model_ckpt_id=79999
```

### RMT + Expert
```
# terminal 0
uv run scripts/serve_policy.py --seed=7  --port=8001 policy:checkpoint --policy.dir=runs/ckpts/mme_vla_suite/recurrent-rmt-expert/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
python examples/robomme/eval.py --args.model_seed=7 --args.port=8001 --args.policy_name=recurrent-rmt-expert --args.model_ckpt_id=79999
```


## Other Hints
You can eval only for a subset of tasks 
```
python examples/robomme/eval.py --args.only_tasks="BinFill,PickXtimes" ...
```
You can exclude or re-eval with `--args.exclude_tasks` and `--args.re_eval_tasks`
Everything, you just rerun the `python examples/robomme/eval.py`, the evalution will automatically resume.