import torch
import time
from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

# pipeline.pipeline.unet.model = torch.compile(
#     pipeline.pipeline.unet.model, mode="max-autotune-no-cudagraphs"
# )
print("Warmup")
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
).images[0]
print("Inference")
st = time.perf_counter()
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
).images[0]
if distri_config.rank == 0:
    print("Inference Time:", time.perf_counter() - st)
    image.save("astronaut.png")
