import torch
from .models.distri_sdxl_unet_pp import DistriUNetPP
from .utils import DistriConfig, PatchParallelismCommManager
from diffusers import DiffusionPipeline, UNet2DConditionModel
from inspect import signature
from distrifuser.modules.pp.attn import DistriCrossAttentionPP, DistriSelfAttentionPP
from distrifuser.modules.base_module import BaseModule
from distrifuser.modules.pp.conv2d import DistriConv2dPP
from distrifuser.modules.pp.groupnorm import DistriGroupNorm


class DistrifuserPipeline:
    def __init__(self, pipeline: DiffusionPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config
        self.static_inputs = None

        if isinstance(pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = DistriUNetPP(self.pipeline.unet, module_config)

        self.prepare()

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if not "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.pipeline.unet.set_counter(0)
        return self.pipeline(*args, **kwargs)

    def encode_prompts(self, pipeline, **kwargs):
        return pipeline.encode_prompt(
            **{k: v for k, v in kwargs.items() if k in signature(pipeline.encode_prompt).parameters}
        )

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        device = distri_config.device

        kwargs["prompt"] = ""
        kwargs["prompt_2"] = None
        kwargs["device"] = device
        kwargs["num_images_per_prompt"] = 1
        kwargs["do_classifier_free_guidance"] = False
        kwargs["negative_prompt"] = None
        kwargs["negative_prompt_2"] = None
        kwargs["prompt_embeds"] = None
        kwargs["negative_prompt_embeds"] = None
        kwargs["pooled_prompt_embeds"] = None
        kwargs["negative_pooled_prompt_embeds"] = None
        kwargs["lora_scale"] = None
        kwargs["clip_skip"] = None

        pooled_prompt_embeds = None
        if getattr(pipeline, "text_encoder_2", False):
            prompt_embeds = torch.rand((1, 77, 2048), dtype=pipeline.dtype).to(device)
            pooled_prompt_embeds = torch.rand((1, 1280), dtype=pipeline.dtype).to(device)
        else:
            prompt_embeds = torch.rand((1, 77, 768), dtype=pipeline.dtype).to(device)

        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        num_channels_latents = pipeline.unet.config.in_channels

        latents = pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            None,
        )
        add_text_embeds, add_time_ids = torch.tensor([1]), torch.tensor([1])
        if pooled_prompt_embeds is not None or getattr(pipeline, "text_encoder_2", False):
            # 7. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            if pipeline.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

            add_time_ids = pipeline._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1, 1)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            add_text_embeds = add_text_embeds.repeat(batch_size, 1)
            add_time_ids = add_time_ids.repeat(batch_size, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["sample"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds
        if pooled_prompt_embeds is not None:
            static_inputs["added_cond_kwargs"] = added_cond_kwargs
        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.unet.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.unet.set_counter(0)
            pipeline.unet(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        pipeline.unet.set_counter(0)
        pipeline.unet(**static_inputs, return_dict=False, record=True)


        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [
                    0,
                    distri_config.warmup_steps + 1,
                    distri_config.warmup_steps + 2,
                ]
            elif distri_config.parallelism == "tensor":
                counters = [0]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            for counter in counters:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pipeline.unet.set_counter(counter)
                    output = pipeline.unet(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                cuda_graphs.append(graph)
            pipeline.unet.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs

    @staticmethod
    def update_distri_config(cls, distri_config):
        cls.distri_config = distri_config
        cls.pipeline.unet.buffer_list = None
        cls.pipeline.unet.output_buffer = None
        cls.pipeline.unet.static_inputs = None
        cls.pipeline.unet.comm_manager = None
        cls.pipeline.unet.set_run2(cls.pipeline.unet)
        cls.pipeline.unet.update_distri_config(distri_config)
        cls.pipeline.unet.buffer_list = None
        cls.pipeline.unet.output_buffer = None

        for name, module in cls.pipeline.unet.named_modules():
            if isinstance(module, BaseModule):
                continue
            for subname, submodule in module.named_children():
                if isinstance(submodule, DistriConv2dPP):
                    submodule.set_run2(submodule)
                    submodule.update_distri_config(distri_config)
                    submodule.buffer_list = None
                    submodule.comm_manager.buffer_list = None
                if isinstance(submodule, DistriSelfAttentionPP):
                    submodule.set_run2(submodule)
                    submodule.update_distri_config(distri_config)
                    submodule.buffer_list = None
                    submodule.comm_manager.buffer_list = None
                if isinstance(submodule, DistriCrossAttentionPP):
                    submodule.set_run2(submodule)
                    submodule.update_distri_config(distri_config)
                    submodule.buffer_list = None
                    submodule.comm_manager.buffer_list = None
                elif isinstance(submodule, DistriGroupNorm):
                    submodule.set_run2(submodule)
                    submodule.update_distri_config(distri_config)
                    submodule.buffer_list = None
                    submodule.comm_manager.buffer_list = None
        cls.prepare()


def parallelize_pipe(pipe: DiffusionPipeline, **kwargs):
    if not getattr(pipe, "_is_parallelized", False):
        distri_config = DistriConfig(kwargs["height"], kwargs["width"], warmup_steps=4, mode="stale_gn")

        pipe = DistrifuserPipeline(pipe, distri_config)

        pipe.__class__._is_parallelized = True
    else:
        if isinstance(pipe, DistrifuserPipeline):
            distri_config = DistriConfig(kwargs["height"], kwargs["width"], warmup_steps=4, mode="stale_gn")
            pipe.update_distri_config(pipe, distri_config=distri_config)

    return pipe
