# MME-VLA Policy Learning and Evaluation



-------
## Updates

- [x] We realease MME-VLA Suite! A family xxx

-------
## Installation
```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```


```
uv venv --python 3.11 examples/history_bench_sim/.venv
uv pip install -e packages/openpi-client
uv pip install -e third_party/maniskill
uv pip install -e  historybench
uv pip install -r examples/history_bench_sim/requirement.txt
```


```
# micromamba create -n qwen_vl python=3.11
# micromamba install flash-attn=2.8.3 flash-attn-fused-dense=2.8.3 flash-attn-layer-norm=2.8.3
# pip install "transformers>=4.57" "qwen_vl_utils>=0.0.14" "ms-swift>=3.9.1" google-generativeai==0.8.6 torchvision==0.24.1 deepspeed==0.18.3

micromamba create -f env.yaml
# pip install -e packages/openpi-client
# pip install -e third_party/ManiSkill
# pip install -e /home/daiyp/openpi/third_party/HistoryBench
```

## Download ckepoint

set OPENPI_DATA_HOME

```
from openpi.shared import download
OPENPI_DATA_HOME = os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")
download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
```

download vision encoder ckpt into OPENPI_DATA_HOME `cd $OPENPI_DATA_HOME` and downloade the seperate [pi05_vision_encoder](`https://huggingface.co/Yinpei/pi05_vision_encoder`)
this is used for dataset feature construction without loading the whole pi05 model.

download qwen3-vl for VLM subgoal training, we use lora training for qwen3-vl.
```
cd runs/ckpts/vlm_subgoal_predictor
git clone git@hf.co:Qwen/Qwen3-VL-4B-Instruct
```

# Data


## experiments

`runs`

Our code mirrors the original openpi structure, we try our best not touch the internal code of openpi, except a few parts, so it can seamless intergrated with pi05


Download data
raw / preprocessed

run preprocess needs 2-3 hours


get norm stats
```
uv run scripts/compute_norm_stats.py
```




## train VLM predictor
```
micromamba activate robomme
bash scripts/finetune_vlm_subgoal_predictor.sh
```

set up google api key `GOOGLE_API_KEY` 


Questions:

1. Why the instructions has slight different?

PatternLock/DrawPattern

VLM gournded subgoal, why the bbox range is different?
it's an early mistake, but does not hurt the prediciton.