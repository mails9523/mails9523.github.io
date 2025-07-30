# Copyright 2024 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import logging
from diffusers.models.activations import get_activation
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
from diffusers.models.unets.unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from diffusers.models.unets.unet_3d_condition import UNet3DConditionOutput

import ipdb


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class I2VGenXLTransformerTemporalEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        activation_fn: str = "geglu",
        upcast_attention: bool = False,
        ff_inner_dim: Optional[int] = None,
        dropout: int = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=True,
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=ff_inner_dim,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        ff_output = self.ff(hidden_states, scale=1.0)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class I2VGenXLUNet(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    I2VGenXL UNet. It is a conditional 3D UNet model that takes a noisy sample, conditional state, and a timestep
    and returns a sample-shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 64): Attention head dim.
        num_attention_heads (`int`, *optional*): The number of attention heads.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        norm_num_groups: Optional[int] = 32,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 64,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
    ):
        super().__init__()

        # When we first integrated the UNet into the library, we didn't have `attention_head_dim`. As a consequence
        # of that, we used `num_attention_heads` for arguments that actually denote attention head dimension. This
        # is why we ignore `num_attention_heads` and calculate it from `attention_head_dims` below.
        # This is still an incorrect way of calculating `num_attention_heads` but we need to stick to it
        # without running proper depcrecation cycles for the {down,mid,up} blocks which are a
        # part of the public API.
        num_attention_heads = attention_head_dim

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels + in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=num_attention_heads,
            in_channels=block_out_channels[0],
            num_layers=1,
            norm_num_groups=norm_num_groups,
        )

        # image embedding
        self.image_latents_proj_in = nn.Sequential(
            nn.Conv2d(4, in_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels * 4, in_channels * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels * 4, in_channels, 3, stride=1, padding=1),
        )
        self.image_latents_temporal_encoder = I2VGenXLTransformerTemporalEncoder(
            dim=in_channels,
            num_attention_heads=2,
            ff_inner_dim=in_channels * 4,
            attention_head_dim=in_channels,
            activation_fn="gelu",
        )
        self.image_latents_context_embedding = nn.Sequential(
            nn.Conv2d(4, in_channels * 8, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(in_channels * 8, in_channels * 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels * 16, cross_attention_dim, 3, stride=2, padding=1),
        )

        # other embeddings -- time, context, fps, etc.
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn="silu")
        self.context_embedding = nn.Sequential(
            nn.Linear(cross_attention_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, cross_attention_dim * in_channels),
        )
        self.fps_embedding = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.context_T5_clip_embedding =nn.Linear(768, 1024)

        # blocks
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-05,
                resnet_act_fn="silu",
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                downsample_padding=1,
                dual_cross_attention=False,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=1e-05,
            resnet_act_fn="silu",
            output_scale_factor=1,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-05,
                resnet_act_fn="silu",
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                resolution_idx=i,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-05)
        self.conv_act = get_activation("silu")
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel._set_gradient_checkpointing
    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.enable_freeu
    def enable_freeu(self, s1, s2, b1, b2):
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        fps: torch.Tensor,
        image_latents: torch.Tensor,
        image_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[List[torch.Tensor]] = [],
        timestep_cond: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple[torch.FloatTensor]]:
        r"""
        The [`I2VGenXLUNet`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            fps (`torch.Tensor`): Frames per second for the video being generated. Used as a "micro-condition".
            image_latents (`torch.FloatTensor`): Image encodings from the VAE.
            image_embeddings (`torch.FloatTensor`): Projection embeddings of the conditioning image computed with a vision encoder.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_3d_condition.UNet3DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """

        # with torch.autocast(device_type='cuda', dtype=torch.float16):

        batch_size, channels, num_frames, height, width = sample.shape

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass `timesteps` as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        t_emb = self.time_embedding(t_emb, timestep_cond)

        # 2. FPS
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        fps = fps.expand(fps.shape[0])
        fps_emb = self.fps_embedding(self.time_proj(fps).to(dtype=self.dtype))

        # 3. time + FPS embeddings.
        emb = t_emb + fps_emb
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)

        # 4. context embeddings.
        # The context embeddings consist of both text embeddings from the input prompt
        # AND the image embeddings from the input image. For images, both VAE encodings
        # and the CLIP image embeddings are incorporated.
        # So the final `context_embeddings` becomes the query for cross-attention.
        if len(encoder_hidden_states) > 1:
            clip_output, T5_output= encoder_hidden_states[0], encoder_hidden_states[1]
            transformed_T5_output = self.context_T5_clip_embedding(T5_output)
            context_output = clip_output + transformed_T5_output
            context_emb = sample.new_zeros(batch_size, 0, self.config.cross_attention_dim)
            context_emb = torch.cat([context_emb, context_output], dim=1)
        else:
            context_emb = sample.new_zeros(batch_size, 0, self.config.cross_attention_dim)
            context_emb = torch.cat([context_emb, encoder_hidden_states[0]], dim=1)



        image_latents_for_context_embds = image_latents[:, :, :1, :]
        image_latents_context_embs = image_latents_for_context_embds.permute(0, 2, 1, 3, 4).reshape(
            image_latents_for_context_embds.shape[0] * image_latents_for_context_embds.shape[2],
            image_latents_for_context_embds.shape[1],
            image_latents_for_context_embds.shape[3],
            image_latents_for_context_embds.shape[4],
        )
        image_latents_context_embs = self.image_latents_context_embedding(image_latents_context_embs)

        _batch_size, _channels, _height, _width = image_latents_context_embs.shape
        image_latents_context_embs = image_latents_context_embs.permute(0, 2, 3, 1).reshape(
            _batch_size, _height * _width, _channels
        )
        context_emb = torch.cat([context_emb, image_latents_context_embs], dim=1)
        print(context_emb.shape)
        exit()

        image_emb = self.context_embedding(image_embeddings)
        image_emb = image_emb.view(-1, self.config.in_channels, self.config.cross_attention_dim)
        context_emb = torch.cat([context_emb, image_emb], dim=1)
        context_emb = context_emb.repeat_interleave(repeats=num_frames, dim=0)

        image_latents = image_latents.permute(0, 2, 1, 3, 4).reshape(
            image_latents.shape[0] * image_latents.shape[2],
            image_latents.shape[1],
            image_latents.shape[3],
            image_latents.shape[4],
        )
        image_latents = self.image_latents_proj_in(image_latents)
        image_latents = (
            image_latents[None, :]
            .reshape(batch_size, num_frames, channels, height, width)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch_size * height * width, num_frames, channels)
        )
        image_latents = self.image_latents_temporal_encoder(image_latents)
        image_latents = image_latents.reshape(batch_size, height, width, num_frames, channels).permute(0, 4, 3, 1, 2)

        # 5. pre-process
        sample = torch.cat([sample, image_latents], dim=1)
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = self.conv_in(sample)
        sample = self.transformer_in(
            sample,
            num_frames=num_frames,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # print(sample.shape)
        # print(context_emb.shape)
        # exit()

        # 6. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=context_emb,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        # 7. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=context_emb,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        # 8. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=context_emb,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )

        # 9. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

class ImageLatentsTemporalEncoderProcessor(nn.Module):
    def __init__(
        self,
        hidden_size = 4,
        cross_attention_dim: Optional[int] = None,
        rank: int = 4,
        network_alpha: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # self.hidden_size = hidden_size
        # self.cross_attention_dim = cross_attention_dim
        # self.rank = rank

        # q_rank = kwargs.pop("q_rank", None)
        # q_hidden_size = kwargs.pop("q_hidden_size", None)
        # q_rank = q_rank if q_rank is not None else rank
        # q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        # v_rank = kwargs.pop("v_rank", None)
        # v_hidden_size = kwargs.pop("v_hidden_size", None)
        # v_rank = v_rank if v_rank is not None else rank
        # v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        # out_rank = kwargs.pop("out_rank", None)
        # out_hidden_size = 4
        # out_rank = out_rank if out_rank is not None else rank
        # out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        # self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        # self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        # self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        # self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        # self_cls_name = self.__class__.__name__
        # attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
        # attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
        # attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
        # attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        # attn._modules.pop("processor")
        # attn.processor = AttnProcessor()
        # return attn.processor(attn, hidden_states, *args, **kwargs)
        return hidden_states

def cnt_net_params(net):

    count_param = []
    for name, params in net.named_parameters():
        # if name in open_params:
        count_param.append(params)
    net_params = sum(x.numel() for x in count_param)
    print(f"Paramas: {round(net_params / (1024 * 1024 * 1024), 3)} B")

    
if __name__ == "__main__":
    unet = I2VGenXLUNet(
        sample_size=32,
    )

    cnt_net_params(unet)
    exit()
    
    noisy_latents = torch.rand((1, 4, 8, 56, 56))

    timesteps = torch.tensor([8])

    encoder_hidden_states = torch.rand((1, 77, 1024))

    fps_tensor = torch.tensor([1])

    image_latents = torch.rand((1, 4, 8, 56, 56))

    image_embeddings = torch.rand((1, 1, 1024))

#     tune_mid_block = True
#     tune_up_block = False
#     tune_down_block = True

#     ipdb.set_trace()
    

#     from diffusers.models.attention_processor import LoRAAttnProcessor
#     from diffusers.loaders import AttnProcsLayers

#     # transformer_in.transformer_blocks.0.attn1.processor
#     # transformer_in.transformer_blocks.0.attn2.processor
#     # image_latents_temporal_encoder.attn1.processor
#     # down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor
# #    unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1 

#     lora_attn_procs = {}
#     for name in unet.attn_processors.keys():
#         # cross_attention_dim = unet.config.cross_attention_dim if name.endswith("attn2.processor") else None

#         if 'temp_attentions' not in name and "attn2.processor" in name:
#             cross_attention_dim = unet.config.cross_attention_dim
#         else:
#             cross_attention_dim = None

#         if name.startswith("mid_block"):
#             hidden_size = unet.config.block_out_channels[-1]
#             if not tune_mid_block: 
#                 lora_attn_procs[name] = ImageLatentsTemporalEncoderProcessor()
#                 continue
#         elif name.startswith("up_blocks"):
#             block_id = int(name[len("up_blocks.")])
#             hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#             if not tune_up_block: 
#                 lora_attn_procs[name] = ImageLatentsTemporalEncoderProcessor()
#                 continue
#         elif name.startswith("down_blocks"):
#             block_id = int(name[len("down_blocks.")])
#             hidden_size = unet.config.block_out_channels[block_id]
#             if not tune_down_block: 
#                 lora_attn_procs[name] = ImageLatentsTemporalEncoderProcessor()
#                 continue
#         elif name.startswith("transformer_in"):
#             # hidden_size = 512
#             lora_attn_procs[name] = ImageLatentsTemporalEncoderProcessor()
#             continue
#         elif name.startswith("image_latents_temporal_encoder"):
#             lora_attn_procs[name] = ImageLatentsTemporalEncoderProcessor()
#             continue
#             # print(name)

#         lora_attn_procs[name] = LoRAAttnProcessor(
#             hidden_size=hidden_size,
#             cross_attention_dim=cross_attention_dim,
#             rank=4,
#         )
#     unet.set_attn_processor(lora_attn_procs)
#     lora_layers = AttnProcsLayers(unet.attn_processors)

#     from diffusers.utils import (
#         check_min_version,
#         convert_state_dict_to_diffusers,
#         convert_unet_state_dict_to_peft,
#         is_wandb_available,
#     )


#     ipdb.set_trace()
    
#     # from diffusers.loaders import LoraLoaderMixin

#     # LoraLoaderMixin.save_lora_weights(
#     #     'dir',
#     #     unet_lora_layers=lora_layers
#     #     # text_encoder_lora_layers=text_encoder_lora_layers_to_save,
#     # )

#     from diffusers.loaders import LoraLoaderMixin
#     save_model = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/xxxx/code/i2vgen-xl-diffusers/sd-model-finetuned/logs_i2xgen_lora/checkpoint-20'
#     lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(save_model)
#     ipdb.set_trace()


#     miss, unexp = unet.load_state_dict(lora_state_dict, strict=False)

    model_pred = unet(
        noisy_latents, 
        timesteps, 
        encoder_hidden_states = encoder_hidden_states,
        fps = fps_tensor,
        image_latents = image_latents,
        image_embeddings=image_embeddings,
        # torch.cat([first_image_features, encoder_hidden_states], dim=1)
        ).sample  # torch.Size([1, 4, 8, 28, 28])

    ipdb.set_trace()

    # lora_layers_params = sum(x.numel() for x in lora_layers.parameters())