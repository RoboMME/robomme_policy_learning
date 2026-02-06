import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from openpi.models import siglip as _siglip
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import einops
import openpi.shared.array_typing as at
import os


class SigLipTokenizer:
    def __init__(self, rngs: nnx.Rngs = nnx.Rngs(2), inference_batch_size: int = 64):
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=2048,  # fixed for pi05
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm="bfloat16",
            )
        )
        img.lazy_init(jax.numpy.ones((1, 224, 224, 3)), train=False, rngs=rngs)

        OPENPI_DATA_HOME = os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")

        with open(os.path.join(OPENPI_DATA_HOME, "pi05_vision_encoder", "siglip_params.pkl"), "rb") as f:
            siglip_params = pickle.load(f)

        siglip_params_jnp = jax.tree.map(lambda x: jnp.array(x), siglip_params)
        graph_def, state = nnx.split(img)
        state.replace_by_pure_dict(siglip_params_jnp)

        self.img = nnx.merge(graph_def, state)
        self.inference_batch_size = inference_batch_size

    def __call__(self, x: at.Float[at.Array, "k v 224 224 3"]):
        shape = x.shape
        assert shape[-3:] == (
            224,
            224,
            3,
        ), f"x.shape: {shape}, should be (B, 224, 224, 3)"
        
        x = einops.rearrange(x, "k v h w c -> (k v) h w c")
        out, _ = self.img(x, train=False)
        out = einops.rearrange(out, "(k v) p d -> k v p d", k=shape[0], v=shape[1])
        return out
    
    
    def batch_call(self, x: at.Float[at.Array, "k v 224 224 3"]):
        shape = x.shape
        assert shape[-3:] == (
            224,
            224,
            3,
        ), f"x.shape: {shape}, should be (B, 224, 224, 3)"
        
        batched_x = einops.rearrange(x, "k v h w c -> (k v) h w c")
        total_items = batched_x.shape[0]
        
        outputs = []
        for i in range(0, total_items, self.inference_batch_size):
            batch = batched_x[i:i + self.inference_batch_size]
            out, _ = self.img(batch, train=False)
            outputs.append(out)
        
        out = jnp.concatenate(outputs, axis=0)
        out = einops.rearrange(out, "(k v) p d -> k v p d", k=shape[0], v=shape[1])
        return out


if __name__ == "__main__":
    from PIL import Image
    import pickle
    tokenizer = SigLipTokenizer()
    
    img_path = "asset/robomme_bench.jpg"
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = img.convert("RGB")
    img = np.array(img)
    
    img = img / 255.0 * 2.0 - 1.0
    img = img.reshape(1, 1, 224, 224, 3)
    img = jnp.array(img)
    image_tokens = tokenizer(img) # (1, 1, 256, 2048)
    
    import pdb; pdb.set_trace()

    
