# 🚀 MME-VLA Policy Learning and Evaluation

### [Website](https://robomme.github.io/) | [Paper](https://arxiv.org/abs/2603.04639) | [Benchmark Repo](https://github.com/RoboMME/robomme_benchmark) | [Dataset](https://huggingface.co/datasets/Yinpei/robomme_data_h5) | [Models](https://huggingface.co/Yinpei/mme_vla_suite) | [Leaderboard](https://robomme.github.io/leaderboard.html)

## 🧭 Outline

- [Updates](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#%EF%B8%8F-updates)
- [Installation](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-installation)
  - [Install with uv](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-install-with-uv)
  - [Install with docker](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-install-with-docker)
- [QuickStart](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-quickstart)
- [Repository Structure](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#%EF%B8%8F-repository-structure)
- [Download](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#%EF%B8%8F-download)
  - [Download Training Data](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-download-training-data)
  - [Download Pre-trained Models](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-download-pre-trained-models)
  - [Download Fine-tuned VLA/VLM Checkpoints](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-download-fine-tuned-vlavlm-checkpoints-optional)
- [Model Training](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#%EF%B8%8F-model-training)
  - [Data Preparation](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-data-preparation)
  - [Train π₀.₅ baseline](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#%EF%B8%8F-train-%CF%80-baseline)
  - [Train MME-VLA policies](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-train-mme-vla-policies)
  - [Train VLM subgoal predictor](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-train-vlm-subgoal-predictor)
- [Evaluation](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-evaluation)
  - [Evaluation with the integrated script](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-evaluation-with-the-integrated-script)
  - [Manual evaluation (per model)](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#%EF%B8%8F-manual-evaluation-per-model)
- [RoboMME Challenge Example](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-robomme-challenge-example)
- [Troubleshooting](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#%EF%B8%8F-troubleshooting)
- [Acknowledgement](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-acknowledgement)
- [Citation](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#-citation)

## 🗞️ Updates

- [04/2026] We improve our dataloader for faster data loading, see more details [here]()
- [03/2026] We use MME-VLA (FrameSamp+Modul) as an example for RoboMME Challenge @ CVPR2026 Submission. Please see [here](https://github.com/RoboMME/robomme_policy_learning?tab=readme-ov-file#robomme-challenge-example) for more details.
- [03/2026] We provide MME-VLA as a submission example for CVPR RoboMME Challenge. More details can be found [here](#robomme-challenge).
- [03/2026] 🚀 We release MME-VLA Suite, a family of memory-augmented vision-language-action (VLA) models based on the $\pi_{0.5}$ backbone. See our [paper](https://arxiv.org/abs/2603.04639) and [leaderboard](https://robomme.github.io/leaderboard.html) for more details and analysis.


## 📦 Installation

### 🧩 Install with UV
#### 📥 Install Policy Learning Repo
```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Set the `OPENPI_DATA_HOME` path in your `~/.bashrc`, e.g. `export OPENPI_DATA_HOME=<your_openpi_homedir>`. For more details, please refer to [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main?tab=readme-ov-file#fine-tuned-models).


#### 🎮 Install RoboMME Simulator
Clone the RoboMME submodule:
```
git submodule update --init
```

Then install the RoboMME environment following the documentation [here](examples/robomme/readme.md).
We use separate environments for VLA training/inference and the RoboMME simulator. During evaluation, we use a WebSocket connection between them, following [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main).


### 🐳 Install with Docker
After [downloading the data](#download) in the `data` directory and setting up `runs` in the following [structure](#repository-structure). 
Update the RoboMME submodule with `git submodule update --init`.
Then build the Docker image following [this](docs/docker_installation.md).

## ⚡ QuickStart
### Evaluation
After installing everything correctly, download our best MME-VLA model (i.e., framesamp-modul) from Hugging Face.
```
git clone https://huggingface.co/Yinpei/perceptual-framesamp-modul <your_specify_model_path>
```
Then run
```
# terminal 0
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --seed=7  --port=8000 policy:checkpoint --policy.dir=<your_specify_model_path>/79999 --policy.config=mme_vla_suite

# terminal 1 
micromamba activate robomme
CUDA_VISIBLE_DEVICES=1 python examples/robomme/eval.py --args.model_seed=7 --args.port=8000 --args.policy_name=<your_specify_policy_name> --args.model_ckpt_id=79999
```
Then the evaluations results will be stored in `runs/evaluation/<your_specify_policy_name>/ckpt79999/seed7`
> Remember to manually set CUDA_VISIBLE_DEVICES using one card for serve_policy.py, as JAX will automatically use all GPUs by default.

### Training 
```bash
# Build the assets for MME-VLA and copy the provided norm_stats.json
mkdir -p runs/assets/mme_vla_suite 
cp -r assets/norm_stats.json runs/assets/mme_vla_suite

# Download a small set of preprocessed training dataset for quick training 
mkdir data
git clone git@hf.co:datasets/Yinpei/robomme_preprocessed_data_sample data/robomme_preprocessed_data_sample
# unzip the data file

# Train the MME-VLA model
# Memory Requirement: 4xA40 40GB GPU or 1xH100 80GB GPU
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MME_VLA_TYPE="perceptual-framesamp-modul"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py mme_vla_suite \
--exp-name=<model_name> \
--batch-size=64 \
--num-workers=4 \
--fsdp-devices=4 \
--dataset-path=data/robomme_preprocessed_data_sample \
--model.use_history \
--model.history_config="${MME_VLA_TYPE}.yaml"
```

All possible configs can be found in `src/mme_vla_suite/models/config/robomme`


## 🗂️ Repository Structure
```
.
├── data
│   ├── robomme_h5_data                 # download robomme raw h5 files here
│   └── robomme_preprocessed_data
│   │   ├── data                        # pickle files
│   │   ├── features                    # precompute siglip token embeddings (.npy, default loader)
│   │   ├── features_bin                # repacked .bin files (optional, for bin/mmap loaders)
│   │   ├── meta                        # statistics for robomme
│   │   ├── memer                       # VLM subgoal training data for MemER
│   │   └── qwenvl                      # VLM subgoal training data for QwenVL
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
│   └── openpi                          # original OpenPi code with minor changes
└── third_party
```

This repository is built on top of [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main). We highly recommend becoming familiar with OpenPi first before working with this repo.

## ⬇️ Download

### 🧱 Download Training Data
Place all data under the `data` directory:
```
mkdir data && cd data
```

Download the raw RoboMME training files [here](https://huggingface.co/datasets/Yinpei/robomme_data_h5):
```
git clone git@hf.co:datasets/Yinpei/robomme_data_h5 data/robomme_data_h5
```
Then run 
```
uv run scripts/tarxz_h5.py decompress --input_dir data/robomme_data_h5 --jobs 16 --remove_archive
```
to extract the *.xz files.

(Optional) Download preprocessed RoboMME data [here](https://huggingface.co/datasets/Yinpei/robomme_preprocessed_data):
```
git clone git@hf.co:datasets/Yinpei/robomme_preprocessed_data data/robomme_preprocessed_data
```
and run `uv run scripts/unzip_data.py data/robomme_preprocessed_data` to unzip the files.  

Alternatively, you can run `uv run scripts/build_dataset.py` to generate the preprocessed pickle files (takes about 2–3 hours) and/or the VLM subgoal predictor training data (takes about 30–60 minutes).   

We also provide data in the LeRobot format [here](https://huggingface.co/datasets/Yinpei/robomme_data_lerobot). In our experiments, however, the LeRobot dataloader significantly increased CPU memory usage during training, which can be a bottleneck in shared training environments (e.g., on HPC clusters). For this reason, we use our custom data format and [dataloader](https://github.com/RoboMME/robomme_policy_learning/blob/89efeaab461cc2b00ede344edf4283692e9c3ada/src/mme_vla_suite/training/dataset.py#L42) in this repository. 


### 🧠 Download Pre-trained Models
Download the $\pi_{0.5}$-base backbone:
```
uv run scripts/download_pi05_base.py
```
Download the [pi05_vision_encoder](https://huggingface.co/Yinpei/pi05_vision_encoder), which is a subset of the $\pi_{0.5}$ parameters used for dataset feature construction without loading the full model. Visual token embeddings are computed and cached for training, and the vision encoder remains frozen in our experiment:
```
cd $OPENPI_DATA_HOME
git clone git@hf.co:Yinpei/pi05_vision_encoder
```

### 🧪 Download Fine-tuned VLA/VLM Checkpoints (Optional)
Fine-tuned models and evaluation results are stored under the `runs` directory. Create it if needed:
```
mkdir runs
mkdir runs/ckpts        # save all trained models here
mkdir runs/evaluation   # evaluation results
mkdir runs/assets       # save all normalization statistics files here
```

You can skip the following steps if you plan to fine-tune your own VLA/VLM models directly; see [Model Training](#model-training).

Download MME-VLA variants [here](https://huggingface.co/Yinpei/mme_vla_suite):
```
git clone git@hf.co:Yinpei/mme_vla_suite runs/ckpts/mme_vla_suite
```
We release all checkpoints for symbolic and perceptual memory, and a subset of recurrent memory variants for research. Recurrent memory is still underperforming; we will release more recurrent variants as results improve.

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


## 🏋️ Model Training

### 🧰 Data Preparation
Prepare training data by either downloading [preprocessed files](https://huggingface.co/datasets/Yinpei/robomme_preprocessed_data) or running:
```
uv run scripts/build_dataset.py --dataset_type robomme_pkl --raw_data_path <downloaded_h5_data_dir> --preprocessed_data_path <your_target_dir>
```

Then compute normalization statistics (this takes about 3 minutes):
```
uv run scripts/compute_norm_stats.py --config-name mme_vla_suite --repo-id robomme --dataset-path="data/robomme_preprocessed_data"
uv run scripts/compute_norm_stats.py --config-name pi05_baseline --repo-id robomme --dataset-path="data/robomme_preprocessed_data"
```
This produces the following structure under `runs`:
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

You can also compare against our reference `norm_stats.json` provided [here](assets/norm_stats.json) to check whether your processing is correct. Small differences are acceptable.

### ⚡ (Optional) Fast dataloader — convert features to `.bin`

The default `RoboMMEDataset` loads history features from `features/episode_X/token_emb_*.npy` — one pickled file per frame (≈150 k files per 1600-episode dataset). Each training sample opens and unpickles 5–64 of these files, which becomes the main bottleneck on NFS storage.

We provide two faster alternatives that use the same feature data repacked into per-episode `.bin` files:

| Loader | Backend | Best for |
|---|---|---|
| `bin` | `os.pread` on cached file descriptors | NFS-mounted datasets, strict safety |
| `mmap` | `np.memmap` | Local SSD, fastest overall |

**Storage layout** under `features_bin/` (per resolution, per episode):
```
features_bin/
├── image_emb_{8x8,4x4,2x2}/episode_N.bin   # bf16
├── pos_emb_{8x8,4x4,2x2}/episode_N.bin     # f32
├── state/episode_N.bin                      # f32
├── kept_indices/episode_N.json              # copied for token-dropping mode
└── metadata.json
```

Each training run only reads the one resolution its memory config needs (e.g. `framesamp-modul` → 4x4, `tokendrop-modul`/`recurrent-ttt-modul` → 8x8).

**Conversion (one-time, runs on the existing `features/` directory):**
```bash
uv run scripts/convert_features_to_bin.py \
  --features_dir data/robomme_preprocessed_data/features \
  --output_dir data/robomme_preprocessed_data/features_bin
```

**Measured speedups** on `data/robomme_preprocessed_data_sample` (per-sample `__getitem__` timing, warm page cache):

| History config | `npy` | `bin` | `mmap` |
|---|---|---|---|
| no-history (baseline) | 5.3 ms | 2.4 ms | 2.7 ms |
| perceptual-framesamp-modul (4x4) | 22.9 ms | 5.8 ms | **4.0 ms** |
| perceptual-tokendrop-modul (8x8) | 17.8 ms | **8.2 ms** | 8.9 ms |
| recurrent-ttt-modul (8x8) | 34.8 ms | 25.8 ms | **21.1 ms** |


**Using a new loader during training** — pass `--dataset-type={npy,bin,mmap}` on the CLI. Default is `npy` for backward compatibility:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py mme_vla_suite \
  --exp-name=<model_name> \
  --batch-size=64 --num-workers=4 --fsdp-devices=4 \
  --dataset-path=data/robomme_preprocessed_data \
  --dataset-type=mmap \
  --model.use_history \
  --model.history_config="${MME_VLA_TYPE}.yaml"
```

### 🎛️ Train π₀.₅ baseline
This variant does not use history and fine-tunes the $\pi_{0.5}$ checkpoints with the vision encoder frozen (for comparison with MME-VLA):
```
bash scripts/finetune_pi05_baseline.sh
```
You can change `--exp-name` to suit your own experiment naming.

### 🧠 Train MME-VLA policies
```
bash scripts/finetune_mme_vla_suite.sh
```
Set `MME_VLA_TYPE` to train a specific model variant. You can also change `--exp-name` to suit your own experiment naming.

We provide a sample training-curve description in [`docs/training_curve_sample.md`](docs/training_curve_sample.md).

### 🧭 Train VLM subgoal predictor
[robomme_preprocessed_data](https://huggingface.co/datasets/Yinpei/robomme_preprocessed_data) already contains VLM subgoal prediction data, but you can also generate it with:
```
uv run scripts/build_dataset.py --dataset_type vlm_subgoal_qwenvl  --raw_data_path=<downloaded_h5_data_dir> --preprocessed_data_path=<your_target_dir>
uv run scripts/build_dataset.py --dataset_type vlm_subgoal_memer  --raw_data_path=<downloaded_h5_data_dir> --preprocessed_data_path=<your_target_dir>
```

After the data is ready, run:
```
micromamba activate robomme
bash scripts/finetune_vlm_subgoal_predictor.sh
```
Set `DATASET_PATH` according to which VLM you are training: (1) simple subgoals, (2) grounded subgoals, or (3) MemER-style subgoals.


## 🧪 Evaluation

### 🚀 Evaluation with the integrated script
After downloading the fine-tuned checkpoints, run:
```
bash scripts/eval.sh
```
Set the `MODEL_TYPE` variable to one of the following:
1. **Prior methods:** `pi05_baseline`, `MemER`
2. **Symbolic MME-VLA:** `symbolic_simpleSG_oracle`, `symbolic_simpleSG_gemini`, `symbolic_simpleSG_qwenvl`, `symbolic_groundedSG_oracle`, `symbolic_groundedSG_gemini`, `symbolic_groundedSG_qwenvl`
3. **Perceptual MME-VLA:** `perceptual-framesamp-context`, `perceptual-framesamp-modul`, `perceptual-framesamp-expert`, `perceptual-tokendrop-context`, `perceptual-tokendrop-modul`, `perceptual-tokendrop-expert`
4. **Recurrent MME-VLA:** `recurrent-rmt-context`, `recurrent-rmt-modul`, `recurrent-rmt-expert`, `recurrent-ttt-context`, `recurrent-ttt-modul`, `recurrent-ttt-expert`

Running `eval.sh` automatically starts two tmux windows: one for the policy server and one for RoboMME evaluation. If the evaluation is interrupted, you can rerun the script; it will automatically resume from the generated `progress.json`.

We also provide a selection of MME-VLA model evaluation logs [here](https://huggingface.co/datasets/Yinpei/selected_robomme_eval_results). For more comprehensive logs, please rerun the evaluation following the guidance.

### ✍️ Manual evaluation (per model)
Details are provided [here](docs/manual_evaluation.md).


## 🏆 RoboMME Challenge Example

We provide a policy-serving example in the [`challenge_interface`](challenge_interface) directory for [RoboMME Challenge](https://robomme.github.io/challenge.html) submission.

We offer three ways for model submission:

1. **Docker-based submission**: see details [here](challenge_interface/docs/submission_guidance_docker.md).
2. **Remote API submission**: see details [here](challenge_interface/docs/submission_guidance_remote.md).
3. **GitHub repo submission**: we will git clone your repo, install the environment, and run the policy server on our machine.

We highly recommend that you first fully understand the MME-VLA policy learning pipeline before diving into this section.


## 🛠️ Troubleshooting
Q1: Vulkan installation fails.  
A1: Please refer to the ManiSkill [solution](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan). If it still does not work, we recommend reinstalling the NVIDIA driver and Vulkan packages. We use NVIDIA driver 570.211.01 and Vulkan 1.3.275. You can also switch to CPU rendering:
```
os.environ['SAPIEN_RENDER_DEVICE'] = 'cpu'
os.environ['MUJOCO_GL'] = 'osmesa'
```

Q2: Why does the evaluation stop?  
A2: We observed that, on long-horizon tasks such as VideoPlaceButton, the WebSocket connection can break due to large video frames. If the evaluation process is interrupted, you can rerun `scripts/eval.sh`, and the program will resume based on the generated `progress.json`.

Q3: CUDA runs out of memory when training VLA models.  
A3: You can set the environment variable `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` to allow JAX to use more GPU memory.

## 🙏 Acknowledgement
This work was supported in part by NSF SES-2128623, NSF CAREER #2337870, NSF NRI #2220876, NSF NAIRR250085, and NSF IIS-1949634. We would also like to thank the excellent [OpenPi](https://github.com/Physical-Intelligence/openpi/tree/main) codebase from Physical-Intelligence.


## 📝 Citation

```
@article{dai2026robomme,
  title={RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies},
  author={Dai, Yinpei and Fu, Hongze and Lee, Jayjun and and Liu, Yuejiang and Zhang, Haoran and Yang, Jianing and Finn, Chelsea and Fazeli, Nima and Chai, Joyce},
  journal={arXiv preprint arXiv:2603.04639},
  year={2026}
}
```
