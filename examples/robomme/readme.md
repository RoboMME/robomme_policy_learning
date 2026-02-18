# RoboMME Evaluation

## Install
```
micromamba create -n robomme python=3.11 -y
micromamba activate robomme
pip install -r examples/robomme/requirements.txt 
pip install -e third_party/robomme_benchmark
pip install -e packages/openpi-client
pip install flash-attn==2.8.3 --no-build-isolation # Optional, it may take 20-30 min to build, you can always choose 'sdpa' in transformer attn_impl 
```

Test if robomme install correctly
```
micromamba activate robomme
python examples/robomme/simple_test.py
```


## Structure
```
examples/robomme
├── env_runner.py           # A wrapper of robomme simulator for eval
├── eval.py                 # The main script for eval
├── simple_test.py          # sanity check
├── subgoal_prediction
│   ├── gemini              # Prompts and API calling for Gemini-2.5-Pro
│   └── qwenvl              # scripts for calling fine-tuned Qwen3VL-4B via swift
├── subgoal_predictor.py    # A wrapper for subgoal predictor (Gemini, QwenVL, Oracle)
└── utils.py
```

