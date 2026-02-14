# MME-VLA Policy Learning and Evaluation

## Outline

- [Updates](#updates)
- [Installation](#installation)
  - [Install RoboMME Simulator](#install-robomme-simulator)
  - [Install MME-VLA-Suite Repo](#install-mme-vla-suite-repo)
- [Repository Structure](#repository-structure)
- [Download](#download)
  - [Download Training Data](#download-training-data)
  - [Download Pre-trained Models](#download-pre-trained-models)
  - [Download Fine-tuned VLA/VLM Checkpoints](#download-fine-tuned-vlavlm-checkpoints)
- [Model Training](#model-training)
  - [Data Preparation](#data-preparation)
  - [Train π₀.₅ baseline](#train-π₀₅-baseline)
  - [Train MME-VLA policies](#train-mme-vla-policies)
  - [Train VLM subgoal predictor](#train-vlm-subgoal-predictor)
- [Evaluation](#evaluation)
  - [Evaluation with the integrated script](#evaluation-with-the-integrated-script)
  - [Manual evaluation (per model)](#manual-evaluation-per-model)
- [TroubleShooting](#troubleshooting)
- [Ackowledgement](#ackowledgement)
- [Bibtext](#bibtext)

## Updates

- [02/2026] We release MME-VLA Suite, a family of memory-augmented vision-language-action (VLA) models based on the $\pi_{0.5}$ backbone. See our paper for details and analysis.




## Installation


### Install RoboMME Simulator
Clone the repo and initialize the RoboMME submodules:
```
git clone git@github.com:RoboMME/MME-VLA-Suite.git
cd MME-VLA-Suite
git submodule update
```

Then install the robomme environment following the document [here](examples/robomme/readme.md)

### Install MME-VLA-Suite Repo
```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Set the `OPENPI_DATA_HOME` path in your `~/.bashrc`, e.g. `export OPENPI_DATA_HOME=<your_openpi_homedir>`.

## Repository Structure
```
.
├── data
│   ├── robomme_h5_data                 # download robomme raw files here
│   ├── robomme_preprocessed_data       # preprocessed robomme data
│   └── vlm_subgoal_prediction_data     # subgoal data for VLM prediction, used in symbolic memory
├── examples
│   └── robomme                         # RoboMME simulator codes
├── packages
├── runs
│   ├── assets                          # save norm_stats json files
│   ├── ckpts                           # fine-tuned checkpoints
│   └── evaluation                      # evaluation results
├── scripts                             # train/eval/data generation scripts
├── setup_robomme.bash
├── src
│   ├── mme_vla_suite                   # MME_VLA code, follows openpi structure 
│   └── openpi                          # original openpi code with minor changes
└── third_party
```


## Download

### Download Training Data
Place all data under the `data` directory:
```
cd MME-VLA-Suite
mkdir data && cd data
```

Download the raw RoboMME training files [here](https://huggingface.co/datasets/Yinpei/robomme_h5_data):
```
git clone git@hf.co:datasets/Yinpei/robomme_h5_data
```

**(Optional)** Download preprocessed RoboMME data [here](https://huggingface.co/datasets/Yinpei/robomme_preprocessed_data):
```
git clone git@hf.co:datasets/Yinpei/robomme_preprocessed_data
```
Alternatively, run `uv run scripts/build_robomme_dataset.py` to generate the preprocessed data (takes about 2–3 hours).


**(Optional)** Download VLM subgoal prediction training data [here](https://huggingface.co/datasets/Yinpei/vlm_subgoal_prediction_data):
```
git clone git@hf.co:datasets/Yinpei/vlm_subgoal_prediction_data
```
Alternatively, run `uv run scripts/build_vlm_subgoal_dataset.py` and `uv run scripts/build_vlm_subgoal_dataset_memer.py` to generate them (takes about 30 minutes).



### Download Pre-trained Models
Download the $\pi_{0.5}$-base backbone:
```
from openpi.shared import download
OPENPI_DATA_HOME = os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")
download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
```
Download the [pi05_vision_encoder](https://huggingface.co/Yinpei/pi05_vision_encoder) (a subset of the $\pi_{0.5}$ parameters used for dataset feature construction without loading the full model; visual token embeddings are computed and cached for training, and the vision encoder stays frozen):
```
cd $OPENPI_DATA_HOME
git clone git@hf.co:Yinpei/pi05_vision_encoder
```

Download [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), used for VLM subgoal prediction with symbolic memory in MME-VLA:
```
cd runs/ckpts/vlm_subgoal_predictor
git clone git@hf.co:Qwen/Qwen3-VL-4B-Instruct
```

### Download Fine-tuned VLA/VLM Checkpoints
Fine-tuned models and evaluation results are stored under the `runs` directory. Create it if needed:
```
cd MME-VLA-Suite
mkdir runs
mkdir runs/ckpts        # save all trained models here
mkdir runs/evaluation   # evaluation results
mkdir runs/assets       # save all normalization statistics files here
```

Download MME-VLA variants [here](https://huggingface.co/Yinpei/mme_vla_suite):
```
cd MME-VLA-Suite/runs/ckpts
git clone git@hf.co:Yinpei/mme_vla_suite
```
We release all checkpoints for symbolic and perceptual memory, and a subset of recurrent memory for research. Recurrent memory is still underperforming; we will release more recurrent variants as results improve.

Download VLM subgoal predictors [here](https://huggingface.co/Yinpei/vlm_subgoal_predictor):
```
cd MME-VLA-Suite/runs/ckpts
git clone git@hf.co:Yinpei/vlm_subgoal_predictor
```

Download the fine-tuned $\pi_{0.5}$ baseline [here](https://huggingface.co/Yinpei/pi05_baseline):
```
cd MME-VLA-Suite/runs/ckpts
git clone git@hf.co:Yinpei/pi05_baseline
```



## Model Training

### Data Preparation
Prepare training data by either downloading [preprocessed files](https://huggingface.co/datasets/Yinpei/robomme_preprocessed_data) or running:
```
uv run scripts/build_robomme_dataset.py --raw_data_path="data/robomme_h5_data" --preprocessed_data_path="data/robomme_preprocessed_data"
```

Then compute normalization statistics (takes about 30 minutes):
```
uv run compute_norm_stats.py --config-name mme_vla_suite --repo-id robomme --dataset-path="data/robomme_preprocessed_data"
uv run compute_norm_stats.py --config-name pi05_baseline --repo-id robomme --dataset-path="data/robomme_preprocessed_data"
```
This produces the following under `runs`:
```
.
├── assets
│   ├── mme_vla_suite
│   │   └── robomme
│   │       └── norm_stats.json
│   └── pi05_baseline
│       └── robomme
│           └── norm_stats.json
```

### Train π₀.₅ baseline
This variant uses no history and fine-tunes the $\pi_{0.5}$ checkpoints with the vision encoder frozen (for comparison with MME-VLA):
```
bash scripts/finetune_pi05_baseline.sh
```


### Train MME-VLA policies
```
bash scripts/finetune_mme_vla_suite.sh
```
Set `MME_VLA_TYPE` to train a specific model variant.

### Train VLM subgoal predictor
Download the VLM subgoal prediction [data](https://huggingface.co/datasets/Yinpei/vlm_subgoal_prediction_data), or generate it with `uv run scripts/build_vlm_subgoal_dataset.py` and `uv run scripts/build_vlm_subgoal_dataset_memer.py`.

```
bash scripts/finetune_vlm_subgoal_predictor.sh
```
Set `DATASET_PATH` according to which VLM you are training: (1) simple subgoals, (2) grounded subgoals, or (3) MemER-style subgoals.


## Evaluation

### Evaluation with the integrated script
After downloading the fine-tuned checkpoints, run:
```
bash scripts/eval.sh
```
Set the `MODEL_TYPE` variable to one of the following:
1. **Prior methods (2):** `pi05_baseline`, `MemER`
2. **Symbolic MME-VLA (6):** `symbolic_simpleSG_oracle`, `symbolic_simpleSG_gemini`, `symbolic_simpleSG_qwenvl`, `symbolic_groundedSG_oracle`, `symbolic_groundedSG_gemini`, `symbolic_groundedSG_qwenvl`
3. **Perceptual MME-VLA (6):** `perceptual-framesamp-context`, `perceptual-framesamp-modul`, `perceptual-framesamp-expert`, `perceptual-tokendrop-context`, `perceptual-tokendrop-modul`, `perceptual-tokendrop-expert`
4. **Recurrent MME-VLA (6):** `recurrent-rmt-context`, `recurrent-rmt-modul`, `recurrent-rmt-expert`, `recurrent-ttt-context`, `recurrent-ttt-modul`, `recurrent-ttt-expert`

Running `eval.sh` automatically starts two tmux windows: one for the policy server and one for RoboMME evaluation. If the evaluation is interrupt, you can re run the script it will automaticaly resume from the genereated `progress.json`.


### Manual evaluation (per model)
Details are provided in [here](manual_evaluation.md)



## TroubleShooting
Q1: Failure about Vulkan Installation.  
A1: We recommend resintall nvidia driver and vulkan packages, we use nvidia driver 570.211.01 and vulkan 1.3.275.  If still does not work, you can switch into CPU rendering
```
os.environ['SAPIEN_RENDER_DEVICE'] = 'cpu'
os.environ['MUJOCO_GL'] = 'osmesa'
```


## Ackowledgement 
This work was supported in part by NSF SES-2128623, NSF CAREER #2337870, NSF NRI #2220876, NSF NAIRR250085. We would also like to thank the wondeful [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main) codebase from Physical-Intelligence.


## Bibtex

```
...
```