import torch
from diffusers.models.attention_processor import Attention
from torch import distributed as dist, nn
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.auraflow_transformer_2d import (
    AuraFlowTransformer2DModel,
)

from .base_model import BaseModel
from distrifuser.modules.pp.attn import (
    DistriSD3AttentionPP,
)

from distrifuser.modules.base_module import BaseModule
from distrifuser.modules.pp.conv2d import DistriConv2dPP, DistriPatchEmbedPP
from distrifuser.modules.pp.groupnorm import DistriGroupNorm
from ..utils import DistriConfig
from diffusers.models.embeddings import PatchEmbed


class DistriAuraTransPP(BaseModel):
    def __init__(self, model: AuraFlowTransformer2DModel, distri_config: DistriConfig):
        assert isinstance(model, AuraFlowTransformer2DModel)

        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, nn.Conv2d):
                        # continue
                        kernel_size = submodule.kernel_size
                        if kernel_size == (1, 1) or kernel_size == 1:
                            continue
                        wrapped_submodule = DistriConv2dPP(
                            submodule, distri_config, is_first_layer=subname == "proj"
                        )
                        setattr(module, subname, wrapped_submodule)
                    if isinstance(submodule, PatchEmbed):
                        # pass
                        wrapped_submodule = DistriPatchEmbedPP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)

                    if isinstance(submodule, Attention):
                        if subname == "attn":  # self attention
                            wrapped_submodule = DistriSD3AttentionPP(
                                submodule, distri_config
                            )
                            setattr(module, subname, wrapped_submodule)

                    elif isinstance(submodule, nn.GroupNorm):
                        wrapped_submodule = DistriGroupNorm(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)

        super(DistriAuraTransPP, self).__init__(model, distri_config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        return_dict: bool = True,  # yes
        record: bool = False,
    ):
        distri_config = self.distri_config
        b, c, h, w = hidden_states.shape

        if distri_config.use_cuda_graph and not record:
            static_inputs = self.static_inputs

            if (
                distri_config.world_size > 1
                and distri_config.do_classifier_free_guidance
                and distri_config.split_batch
            ):
                assert b == 2
                batch_idx = distri_config.batch_idx()
                hidden_states = hidden_states[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1]
                    if torch.is_tensor(timestep) and timestep.ndim > 0
                    else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]

            assert static_inputs["hidden_states"].shape == hidden_states.shape
            static_inputs["hidden_states"].copy_(hidden_states)
            if torch.is_tensor(timestep):
                if timestep.ndim == 0:
                    for b in range(static_inputs["timestep"].shape[0]):
                        static_inputs["timestep"][b] = timestep.item()
                else:
                    assert static_inputs["timestep"].shape == timestep.shape
                    static_inputs["timestep"].copy_(timestep)
            else:
                for b in range(static_inputs["timestep"].shape[0]):
                    static_inputs["timestep"][b] = timestep

            assert (
                static_inputs["encoder_hidden_states"].shape
                == encoder_hidden_states.shape
            )
            static_inputs["encoder_hidden_states"].copy_(encoder_hidden_states)

            if self.counter <= distri_config.warmup_steps:
                graph_idx = 0
            elif self.counter == distri_config.warmup_steps + 1:
                graph_idx = 1
            else:
                graph_idx = 2

            self.cuda_graphs[graph_idx].replay()
            output = self.static_outputs[graph_idx]
        else:
            if distri_config.world_size == 1:
                output = self.model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

            elif (
                distri_config.do_classifier_free_guidance and distri_config.split_batch
            ):
                assert b == 2
                batch_idx = distri_config.batch_idx()
                hidden_states = hidden_states[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1]
                    if torch.is_tensor(timestep) and timestep.ndim > 0
                    else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                output = self.model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty(
                        (b, c, h, w), device=output.device, dtype=output.dtype
                    )
                if self.buffer_list is None:
                    self.buffer_list = [
                        torch.empty_like(output)
                        for _ in range(distri_config.world_size)
                    ]
                dist.all_gather(self.buffer_list, output.contiguous(), async_op=False)

                torch.cat(
                    self.buffer_list[: distri_config.n_device_per_batch],
                    dim=2,
                    out=self.output_buffer[0:1],
                )
                torch.cat(
                    self.buffer_list[distri_config.n_device_per_batch :],
                    dim=2,
                    out=self.output_buffer[1:2],
                )
                output = self.output_buffer

            else:
                output = self.model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty(
                        (b, c, h, w), device=output.device, dtype=output.dtype
                    )
                if self.buffer_list is None:
                    self.buffer_list = [
                        torch.empty_like(output)
                        for _ in range(distri_config.world_size)
                    ]
                output = output.contiguous()
                dist.all_gather(self.buffer_list, output, async_op=False)
                torch.cat(self.buffer_list, dim=2, out=self.output_buffer)
                output = self.output_buffer
            if record:
                if self.static_inputs is None:
                    self.static_inputs = {
                        "hidden_states": hidden_states,
                        "timestep": timestep,
                        "encoder_hidden_states": encoder_hidden_states,
                    }
                self.synchronize()

        if return_dict:
            output = Transformer2DModelOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
