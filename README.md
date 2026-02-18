# MME-VLA Policy Learning and Evaluation

## Outline

- [Updates](#updates)
- [Installation](#installation)
  - [Install MME-VLA-Suite Repo](#install-mme-vla-suite-repo)
  - [Install RoboMME Simulator](#install-robomme-simulator)
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
- [Troubleshooting](#troubleshooting)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Updates

- [02/2026] We release MME-VLA Suite, a family of memory-augmented vision-language-action (VLA) models based on the $\pi_{0.5}$ backbone. See our paper for more details and analysis.


## Installation

### Install Policy Learning Repo
```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Set the `OPENPI_DATA_HOME` path in your `~/.bashrc`, e.g. `export OPENPI_DATA_HOME=<your_openpi_homedir>`. For more details, please refer to [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file#fine-tuned-models).


### Install RoboMME Simulator
Clone the RoboMME submodule:
```
git submodule update --init
```

Then install the RoboMME environment following the documentation [here](examples/robomme/readme.md).
We use separate environments for VLA training/inference and the RoboMME simulator. During evaluation, we use WebSocket to connect them, following [openpi](https://github.com/Physical-Intelligence/openpi/tree/main).

## Repository Structure
```
.
├── data
│   ├── robomme_h5_data                 # download robomme raw files here
│   ├── robomme_preprocessed_data       # preprocessed robomme data
│   └── vlm_subgoal_prediction_data     # subgoal data for VLM prediction, used in symbolic memory
├── examples
│   └── robomme                         # RoboMME simulator evaluation code
├── packages
│   └── openpi-client                   # VLA client & server interface
├── runs
│   ├── assets                          # save norm_stats json files
│   ├── ckpts                           # fine-tuned checkpoints
│   └── evaluation                      # evaluation results
├── scripts                             # train/eval/data_generation scripts
├── src
│   ├── mme_vla_suite                   # MME_VLA code, follows openpi structure 
│   └── openpi                          # original openpi code with minor changes
└── third_party
```

This repo is built on [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main). We highly recommend getting familiar with OpenPi first before working with this repo.

## Download

### Download Training Data
Place all data under the `data` directory:
```
mkdir data && cd data
```

Download the raw RoboMME training files [here](https://huggingface.co/datasets/Yinpei/robomme_h5_data):
```
git clone git@hf.co:datasets/Yinpei/robomme_h5_data data/robomme_h5_data
```

**(Optional)** Download preprocessed RoboMME data [here](https://huggingface.co/datasets/Yinpei/robomme_preprocessed_data):
```
git clone git@hf.co:datasets/Yinpei/robomme_preprocessed_data data/robomme_preprocessed_data
```
and run `uv run scripts/unzip_data.py data/robomme_preprocessed_data` to unzip the files.  
Alternatively, run `uv run scripts/build_robomme_dataset.py` to generate the preprocessed data (takes about 2–3 hours).


**(Optional)** Download VLM subgoal prediction training data [here](https://huggingface.co/datasets/Yinpei/vlm_subgoal_prediction_data):
```
git clone git@hf.co:datasets/Yinpei/vlm_subgoal_prediction_data data/vlm_subgoal_prediction_data
```
and run `uv run scripts/unzip_data.py data/vlm_subgoal_prediction_data` to unzip the files.  
Alternatively, run `uv run scripts/build_vlm_subgoal_dataset_qwenvl.py` and `uv run scripts/build_vlm_subgoal_dataset_memer.py` to generate them (takes about 30 minutes).



### Download Pre-trained Models
Download the $\pi_{0.5}$-base backbone:
```
uv run scripts/download_pi05_base.py
```
Download the [pi05_vision_encoder](https://huggingface.co/Yinpei/pi05_vision_encoder) (a subset of the $\pi_{0.5}$ parameters used for dataset feature construction without loading the full model; visual token embeddings are computed and cached for training, and the vision encoder stays frozen):
```
cd $OPENPI_DATA_HOME
git clone git@hf.co:Yinpei/pi05_vision_encoder
```

### Download Fine-tuned VLA/VLM Checkpoints (Optional)
Fine-tuned models and evaluation results are stored under the `runs` directory. Create it if needed:
```
mkdir runs
mkdir runs/ckpts        # save all trained models here
mkdir runs/evaluation   # evaluation results
mkdir runs/assets       # save all normalization statistics files here
```

You can skip the following steps if you want to fine-tune your own VLA/VLM directly; see [Model Training](#model-training).

Download MME-VLA variants [here](https://huggingface.co/Yinpei/mme_vla_suite):
```
git clone git@hf.co:Yinpei/mme_vla_suite runs/ckpts/mme_vla_suite
```
We release all checkpoints for symbolic and perceptual memory, and a subset of recurrent memory for research. Recurrent memory is still underperforming; we will release more recurrent variants as results improve.

Download VLM subgoal predictors [here](https://huggingface.co/Yinpei/vlm_subgoal_predictor):
```
git clone git@hf.co:Yinpei/vlm_subgoal_predictor runs/ckpts/vlm_subgoal_predictor
```

Download the fine-tuned $\pi_{0.5}$ baseline [here](https://huggingface.co/Yinpei/pi05_baseline):
```
git clone git@hf.co:Yinpei/pi05_baseline runs/ckpts/pi05_baseline
```

After downloading fine-tuned checkpoints, you can run 
```
uv run ./scripts/unzip_ckpt.py runs/ckpts
```
to unzip all of them.


## Model Training

### Data Preparation
Prepare training data by either downloading [preprocessed files](https://huggingface.co/datasets/Yinpei/robomme_preprocessed_data) or running:
```
uv run scripts/build_robomme_dataset.py --raw_data_path="data/robomme_h5_data" --preprocessed_data_path="data/robomme_preprocessed_data"
```

Then compute normalization statistics (takes about 10 minutes):
```
uv run scripts/compute_norm_stats.py --config-name mme_vla_suite --repo-id robomme --dataset-path="data/robomme_preprocessed_data"
uv run scripts/compute_norm_stats.py --config-name pi05_baseline --repo-id robomme --dataset-path="data/robomme_preprocessed_data"
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

You can also compare with our computed `norm_stats.json` provided [here](assets/norm_stats.json) to check if your processing is correct. A small difference is acceptable.

### Train π₀.₅ baseline
This variant uses no history and fine-tunes the $\pi_{0.5}$ checkpoints with the vision encoder frozen (for comparison with MME-VLA):
```
bash scripts/finetune_pi05_baseline.sh
```
You can change `--exp-name` to adapt to your own needs.

### Train MME-VLA policies
```
bash scripts/finetune_mme_vla_suite.sh
```
Set `MME_VLA_TYPE` to train a specific model variant. You can change `--exp-name` to adapt to your own needs.

### Train VLM subgoal predictor
Download the VLM subgoal prediction [data](https://huggingface.co/datasets/Yinpei/vlm_subgoal_prediction_data), or generate it with `uv run scripts/build_vlm_subgoal_dataset_qwenvl.py` and `uv run scripts/build_vlm_subgoal_dataset_memer.py`.

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
1. **Prior methods:** `pi05_baseline`, `MemER`
2. **Symbolic MME-VLA:** `symbolic_simpleSG_oracle`, `symbolic_simpleSG_gemini`, `symbolic_simpleSG_qwenvl`, `symbolic_groundedSG_oracle`, `symbolic_groundedSG_gemini`, `symbolic_groundedSG_qwenvl`
3. **Perceptual MME-VLA:** `perceptual-framesamp-context`, `perceptual-framesamp-modul`, `perceptual-framesamp-expert`, `perceptual-tokendrop-context`, `perceptual-tokendrop-modul`, `perceptual-tokendrop-expert`
4. **Recurrent MME-VLA:** `recurrent-rmt-context`, `recurrent-rmt-modul`, `recurrent-rmt-expert`, `recurrent-ttt-context`, `recurrent-ttt-modul`, `recurrent-ttt-expert`

Running `eval.sh` automatically starts two tmux windows: one for the policy server and one for RoboMME evaluation. If the evaluation is interrupted, you can re-run the script; it will automatically resume from the generated `progress.json`.


### Manual evaluation (per model)
Details are provided [here](docs/manual_evaluation.md).



## Troubleshooting
Q1: Failure with Vulkan installation.  
A1: We recommend reinstalling the NVIDIA driver and Vulkan packages. We use NVIDIA driver 570.211.01 and Vulkan 1.3.275. If it still does not work, you can switch to CPU rendering:
```
os.environ['SAPIEN_RENDER_DEVICE'] = 'cpu'
os.environ['MUJOCO_GL'] = 'osmesa'
```

Q2: Why does the evaluation stop?
A2: We noticed that sometimes, on long-horizon tasks such as VideoPlaceButton, the WebSocket connection breaks due to large video frames. If the evaluation process is interrupted, you can rerun `scripts/eval.sh`, and the program will resume based on the generated `progress.json`.


## Acknowledgement
This work was supported in part by NSF SES-2128623, NSF CAREER #2337870, NSF NRI #2220876, NSF NAIRR250085. We would also like to thank the wonderful [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main) codebase from Physical-Intelligence.


## Citation

```
...
```