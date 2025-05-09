from diffusers import AuraFlowPipeline
import torch
import time

pipeline = AuraFlowPipeline.from_pretrained(
    "fal/AuraFlow", torch_dtype=torch.float16
).to("cuda")
st = time.perf_counter()
image = pipeline(
    prompt="close-up portrait of a majestic iguana with vibrant blue-green scales, piercing amber eyes, and orange spiky crest. Intricate textures and details visible on scaly skin. Wrapped in dark hood, giving regal appearance. Dramatic lighting against black background. Hyper-realistic, high-resolution image showcasing the reptile's expressive features and coloration.",
    height=1024,
    width=1024,
    num_inference_steps=30,
    generator=torch.Generator().manual_seed(666),
    guidance_scale=3.5,
).images[0]
print("Inference Time", time.perf_counter() - st)
image.save("reference.png")
