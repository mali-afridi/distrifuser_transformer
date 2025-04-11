import torch
import time
from distrifuser.distri_aura_pipeline import DistriAuraPipeline
from distrifuser.utils import DistriConfig

torch._dynamo.config.cache_size_limit = 128
torch.cuda.tunable.enable(val=True)
torch.set_float32_matmul_precision("medium")
torch._inductor.config.reorder_for_compute_comm_overlap = True
torch._dynamo.config.automatic_dynamic_shapes = True
distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
pipeline = DistriAuraPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="fal/AuraFlow",
    torch_dtype=torch.float16,
)
pipeline.pipeline.transformer.to(memory_format=torch.channels_last)
# pipeline.pipeline.transformer = torch.compile(
#     pipeline.pipeline.transformer, dynamic=True, mode="max-autotune-no-cudagraphs"
# )
# warmup incase of compilation
image = pipeline(
    prompt="close-up portrait of a majestic iguana with vibrant blue-green scales, piercing amber eyes, and orange spiky crest. Intricate textures and details visible on scaly skin. Wrapped in dark hood, giving regal appearance. Dramatic lighting against black background. Hyper-realistic, high-resolution image showcasing the reptile's expressive features and coloration.",
    # height=1024,
    # width=1024,
    num_inference_steps=30,
    generator=torch.Generator().manual_seed(666),
    guidance_scale=3.5,
).images[0]
st = time.perf_counter()
image = pipeline(
    prompt="close-up portrait of a majestic iguana with vibrant blue-green scales, piercing amber eyes, and orange spiky crest. Intricate textures and details visible on scaly skin. Wrapped in dark hood, giving regal appearance. Dramatic lighting against black background. Hyper-realistic, high-resolution image showcasing the reptile's expressive features and coloration.",
    # height=1024,
    # width=1024,
    num_inference_steps=30,
    generator=torch.Generator().manual_seed(666),
    guidance_scale=3.5,
).images[0]
print("Inference Time", time.perf_counter() - st)
image.save("aura.png")
