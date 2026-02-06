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



## Download ckepoint

set OPENPI_DATA_HOME

```
from openpi.shared import download
download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
```

download vision encoder ckpt into OPENPI_DATA_HOME `cd $OPENPI_DATA_HOME` and downloade the seperate [pi05_vision_encoder](`https://huggingface.co/Yinpei/pi05_vision_encoder`)
this is used for dataset feature construction without loading the whole pi05 model.

# Data


## experiments

`runs`

Our code mirrors the original openpi structure, we try our best not touch the internal code of openpi, except a few parts, so it can seamless intergrated with pi05


Download data
raw / preprocessed


get norm stats
```
uv run scripts/compute_norm_stats.py
```

