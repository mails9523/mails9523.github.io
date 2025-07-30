# 11/02/2023 07:16:40 - INFO - __main__ - DownBlock3D before: torch.Size([8, 1280, 4, 4])
# 11/02/2023 07:16:40 - INFO - __main__ - DownBlock3D after: torch.Size([8, 1280, 4, 4])

# 11/02/2023 07:16:40 - INFO - __main__ - UpBlock3D before: torch.Size([8, 
# 11/02/2023 07:16:40 - INFO - __main__ - UpBlock3D after: torch.Size([8, 1280, 4, 4])

# FO CrossAttnDownBlock3D before: torch.Size([8, 320, 32, 32])
# 2023-11-02 Thursday 07:28:43 train_text_to_video.py write 78: INFO CrossAttnDownBlock3D after: torch.Size([8, 320, 32, 32])

# 2023-11-02 Thursday 07:28:44 train_text_to_video.py write 78: INFO CrossAttnUpBlock3D before: torch.Size([8, 320, 32, 32])
# 2023-11-02 Thursday 07:28:44 train_text_to_video.py write 78: INFO CrossAttnUpBlock3D after: torch.Size([8, 320, 32, 32])

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import ipdb
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D

# Motivation
# 1. 当离散时，同一实体的变化幅度增大，然而当前基于 1x1 的 conv在建模相邻帧关系时难以捕获实体的变化关系      
# 2. 在平滑时，模型仅需要考虑最近的几帧就可以对下一帧有较好的预测，然而当离散时，模型需要考虑更多的帧来进行预测

# Idea
# -> 1: 在建模时序信息时，需要自适应捕获空间变化
# -> 2: 不止需要考虑临近帧，可能需要考虑更远距离的信息

# Design
# -> 从横纵轴为spatio-temporal的二维矩阵中找相关区域


# 1.  自适应不等长窗口 -> 表情离散，需要考虑更长时间范围内的信息
# 2.  自适应注意范围 -> 由于离散性质带来的局部区域便宜程度大 


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        conv_cls = nn.Conv2d 

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = conv_cls(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels


        hidden_states = self.conv(hidden_states)

        return hidden_states


class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        conv_cls = nn.Conv2d 

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = conv_cls(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                if isinstance(self.conv, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.conv(hidden_states, scale)
                else:
                    hidden_states = self.conv(hidden_states)
            else:
                if isinstance(self.Conv2d_0, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.Conv2d_0(hidden_states, scale)
                else:
                    hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016

    Parameters:
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        hidden_states = (hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4))

        identity = hidden_states # N, C, F, H, W
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states

class SpatioTemporalInteractionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        num_attention_layers:  Optional[int] = 1,
        num_attention_heads: Optional[int] = 8,
        attention_head_dim: Optional[int] = 16,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout

        # define transformers blocks for semantic interaction
        self.transformer_blocks = Attention(
                    self.in_dim,
                    heads = self.num_attention_heads,
                    dim_head = self.attention_head_dim
                )

        # define conv layer for detail preserve
        self.conv_layer = TemporalConvLayer(
            in_dim = self.in_dim,
            dropout = self.dropout
        )

        # define linear layer for adaptive fusion
        # self.feed_forward = FeedForward(
            # dim = 2 * self.in_dim,
            # dim_out = self.out_dim
        # )
        self.w1 = torch.nn.Parameter(torch.zeros(1))
        self.w2 = torch.nn.Parameter(torch.ones(1))
        self.register_parameter('w1', self.w1)
        self.register_parameter('w2', self.w2)

    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        f, dim, h, w = hidden_states.shape

        # semantic interaction
        semantic_states = hidden_states[None, :].permute(0, 1, 3, 4, 2)
        semantic_states = semantic_states.reshape(-1, f * h * w, dim)
        semantic_states = self.transformer_blocks(semantic_states)
        semantic_states = semantic_states.reshape(-1, f, h, w, dim).permute(0, 1, 4, 2, 3).squeeze(dim=0) + hidden_states

        # detail preserve
        detail_states = self.conv_layer(hidden_states)

        # adaptive fusion
        # input_linear = torch.cat([semantic_states, detail_states], dim=1).permute(0, 2, 3, 1)
        # output_linear = self.feed_forward(input_linear).permute(0, 3, 1, 2)
        hidden_states = self.w1 * semantic_states + self.w2 * detail_states

        return hidden_states


class SpatioTemporalInteractionLayerDownSample(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        num_attention_layers:  Optional[int] = 1,
        num_attention_heads: Optional[int] = 8,
        attention_head_dim: Optional[int] = 32,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        num_layers: int = 2, 
        ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout

        # define transformers blocks for semantic interaction
        self.downsample = Downsample2D(channels=self.in_dim)
        self.transformer_blocks = Attention(
                    self.in_dim,
                    heads = self.num_attention_heads,
                    dim_head = self.attention_head_dim
                )
        self.upsample = Upsample2D(channels=self.in_dim)

        # define conv layer for detail preserve
        self.conv_layers = nn.ModuleList([
            ResnetBlock2D(
                in_channels=8,
                out_channels=8,
                groups=1,
                temb_channels=None
            ) for _ in range(num_layers)
        ])
        # define linear layer for adaptive fusion
        # self.feed_forward = FeedForward(
            # dim = 2 * self.in_dim,
            # dim_out = self.out_dim
        # )
        self.w1 = torch.nn.Parameter(torch.zeros(1))
        self.w2 = torch.nn.Parameter(torch.ones(1))
        self.register_parameter('w1', self.w1)
        self.register_parameter('w2', self.w2)
        
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_dim, eps=1e-6, affine=True)

    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        f, dim, h, w = hidden_states.shape

        # semantic interaction
        semantic_states = self.downsample(hidden_states)
        _, dim, h_dp, w_dp = semantic_states.shape        
        semantic_states = semantic_states.reshape(-1, num_frames, dim, h_dp, w_dp)
        semantic_states = semantic_states.permute(0, 1, 3, 4, 2)
        semantic_states = semantic_states.reshape(-1, num_frames * h_dp * w_dp, dim)
        semantic_states = self.transformer_blocks(semantic_states)
        semantic_states = semantic_states.reshape(-1, num_frames, h_dp, w_dp, dim).permute(0, 1, 4, 2, 3).reshape(-1, dim, h_dp, w_dp)
        semantic_states = self.upsample(semantic_states, output_size=[h, w]) + hidden_states

        # detail preserve
        batch_size = f // num_frames
        hidden_states = hidden_states.reshape(batch_size, num_frames, dim, h, w)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)  # [batch_size, dim, num_frames, h, w]
        hidden_states = hidden_states.reshape(batch_size * dim, num_frames, h, w)  # [batch_size * dim, num_frames, h, w]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states, temb=None)
        detail_states = hidden_states

        # reshape detail_states to match semantic_states
        detail_states = detail_states.reshape(batch_size, dim, num_frames, h, w)  # [batch_size, dim, num_frames, h, w]
        detail_states = detail_states.permute(0, 2, 1, 3, 4)  # [batch_size, num_frames, dim, h, w]
        detail_states = detail_states.reshape(-1, dim, h, w)  # [batch_size * num_frames, dim, h, w]

        # adaptive fusion
        hidden_states = self.w1 * semantic_states + self.w2 * detail_states
        
        # add group norm to avoid loss is nan
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
  


class SpatioTemporalInteractionLayerDownSample3Weight(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        num_attention_layers:  Optional[int] = 1,
        num_attention_heads: Optional[int] = 8,
        attention_head_dim: Optional[int] = 32,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout

        # define transformers blocks for semantic interaction
        self.downsample = Downsample2D(channels=self.in_dim)
        self.transformer_blocks = Attention(
                    self.in_dim,
                    heads = self.num_attention_heads,
                    dim_head = self.attention_head_dim
                )
        self.upsample = Upsample2D(channels=self.in_dim)

        # define conv layer for detail preserve
        self.conv_layer = TemporalConvLayer(
            in_dim = self.in_dim,
            dropout = self.dropout
        )

        # define linear layer for adaptive fusion
        # self.feed_forward_1 = FeedForward(dim = 2 * self.in_dim, dim_out = 32)
        # self.feed_forward_2 = FeedForward(dim = 32, dim_out = self.in_dim)
        self.w1 = torch.nn.Parameter(torch.zeros(1))
        self.w2 = torch.nn.Parameter(torch.zeros(1))
        self.w3 = torch.nn.Parameter(torch.ones(1))
        self.register_parameter('w1', self.w1)
        self.register_parameter('w2', self.w2)
        self.register_parameter('w3', self.w3)


    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        f, dim, h, w = hidden_states.shape

        # semantic interaction
        semantic_states = self.downsample(hidden_states)
        _, dim, h_dp, w_dp = semantic_states.shape        
        semantic_states = semantic_states.reshape(-1, num_frames, dim, h_dp, w_dp)
        semantic_states = semantic_states.permute(0, 1, 3, 4, 2)
        semantic_states = semantic_states.reshape(-1, num_frames * h_dp * w_dp, dim)
        semantic_states = self.transformer_blocks(semantic_states)
        semantic_states = semantic_states.reshape(-1, num_frames, h_dp, w_dp, dim).permute(0, 1, 4, 2, 3).reshape(-1, dim, h_dp, w_dp)
        semantic_states = self.upsample(semantic_states, output_size=[h, w]) + hidden_states


        # detail preserve
        detail_states = self.conv_layer(hidden_states)

        # adaptive fusion
        # input_linear = torch.cat([semantic_states, detail_states], dim=1).permute(0, 2, 3, 1)
        hidden_states = self.w1 * semantic_states + self.w2 * detail_states + self.w3 * hidden_states
        
        return hidden_states


if __name__=="__main__":
    # tc1 = TemporalConvLayer(
    #     in_dim = 128
    # )
    tc1 = SpatioTemporalInteractionLayerDownSampleFc(
        in_dim = 128
    )
    tc = SpatioTemporalInteractionLayerDownSample(
        in_dim = 128
    )

    input_tensor = torch.rand(16, 128, 4, 4)
    num_frames = 8
    output_linear = tc(input_tensor, num_frames=8)
    print(output_linear.shape)

    tc_params = sum(x.numel() for x in tc.parameters())
    tc1_params = sum(x.numel() for x in tc1.parameters())
    ipdb.set_trace()
