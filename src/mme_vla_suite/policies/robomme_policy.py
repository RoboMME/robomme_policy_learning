import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robomme_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        assert np.abs(image).max() <= 1.0, "Image is not normalized"
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboMMEInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                # "right_wrist_0_rgb": np.zeros_like(base_image), # remove the third image for memory saving
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                # "right_wrist_0_rgb": np.False_,
            },
            # perceptual memory
            "static_image_emb": data.get("static_image_emb", None), # (budget, d1)
            "static_pos_emb": data.get("static_pos_emb", None), # (budget, d2)
            "static_state_emb": data.get("static_state_emb", None), # (budget, d3)
            "static_mask": data.get("static_mask", None), # (budget)
            # recurrent memory
            "recur_image_emb": data.get("recur_image_emb", None), # (max_recur_steps, views, p, d1)
            "recur_pos_emb": data.get("recur_pos_emb", None), # (max_recur_steps, views, p, d2)
            "recur_state_emb": data.get("recur_state_emb", None), # (max_recur_steps, d3)
            "recur_mask": data.get("recur_mask", None), # (max_recur_steps)
            # symbolic memory
            "simple_subgoal": data.get("simple_subgoal", None),
            "grounded_subgoal": data.get("grounded_subgoal", None),
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboMMEOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])} # joint angles + gripper (1 open -1 close)
