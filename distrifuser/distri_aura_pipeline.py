import torch
from diffusers import (
    AuraFlowPipeline,
)

from .models.distri_aura_trans_pp import DistriAuraTransPP

from .utils import DistriConfig, PatchParallelismCommManager
from diffusers.models.transformers.auraflow_transformer_2d import (
    AuraFlowTransformer2DModel,
)


class DistriAuraPipeline:
    def __init__(self, pipeline: AuraFlowPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "fal/AuraFlow"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        transformer = AuraFlowTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            subfolder="transformer",
        ).to(device)

        if distri_config.parallelism == "patch":
            transformer = DistriAuraTransPP(transformer, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = AuraFlowPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            transformer=transformer,
            **kwargs,
        ).to(device)
        return DistriAuraPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if not "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.pipeline.transformer.set_counter(0)
        return self.pipeline(height=config.height, width=config.width, *args, **kwargs)

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

        device = distri_config.device
        batch_size = 1
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipeline.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            negative_prompt_attention_mask=None,
            max_sequence_length=256,
        )

        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        num_channels_latents = pipeline.transformer.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            None,
        )

        prompt_embeds = prompt_embeds.to(device)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["hidden_states"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.transformer.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.transformer.set_counter(0)
            pipeline.transformer(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.transformer.set_counter(0)
        pipeline.transformer(**static_inputs, return_dict=False, record=True)

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
                    pipeline.transformer.set_counter(counter)
                    output = pipeline.transformer(
                        **static_inputs, return_dict=False, record=True
                    )[0]
                    static_outputs.append(output)
                cuda_graphs.append(graph)
            pipeline.transformer.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs
