# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
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
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import complex_multiplication, conjugate, expand_operator, reduce_operator
from direct.nn.recurrent.recurrent import Conv2dGRU, NormConv2dGRU
from direct.nn.types import InitType


class ChannelAttention(nn.Module):
    """Channel Attention Layer (SE-style) inspired by PromptMR's CALayer."""

    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class PromptGenerator(nn.Module):
    """Visual Prompt Generator for EGLAVarNet with Low-Rank Memory Initialization.
    
    This module extracts multi-scale features, applies global channel attention,
    and composes the initial memory matrix M0 using a low-rank decomposition strategy (M0 = U * V^T).
    This enforces a structural prior on the initial state, capturing essential 
    reconstruction patterns while suppressing high-rank noise.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_components: int = 8, rank: int = 16):
        """
        Args:
            in_channels: Input channels (usually 2 for complex image).
            hidden_channels: Dimension of the GLA hidden state (C).
            num_components: Number of learnable memory templates.
            rank: The rank (r) of the memory initialization. r << C.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_components = num_components
        self.rank = rank
        
        # Low-rank Memory Bank: Instead of C*C, we learn U and V bases.
        # Shape: (N_components, C, rank)
        self.u_bank = nn.Parameter(
            torch.randn(num_components, hidden_channels, rank) * 0.02
        )
        self.v_bank = nn.Parameter(
            torch.randn(num_components, hidden_channels, rank) * 0.02
        )
        
        # Multi-scale feature layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Global Cross-scale Channel Attention (112 channels)
        self.global_ca = ChannelAttention(112, reduction=8)
        
        # MLP to predict the composition weights for the bank
        self.mlp = nn.Sequential(
            nn.Linear(112, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_components)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, 2, H, W).
        Returns:
            out: Dynamically composed low-rank initial memory matrix (B, C, C).
        """
        B = x.shape[0]
        
        # 1. Multi-scale feature extraction
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        
        # 2. Extract channel descriptors (GAP)
        v1 = self.gap(f1)
        v2 = self.gap(f2)
        v3 = self.gap(f3)
        
        # 3. Cross-scale concatenation and Global Attention
        v_all = torch.cat([v1, v2, v3], dim=1)
        v_weighted = self.global_ca(v_all).flatten(1)
        
        # 4. Predict weights for the low-rank banks
        weights = self.mlp(v_weighted)  # (B, num_components)
        weights = F.softmax(weights, dim=1)
        
        # 5. Compose mixed U and V bases
        # (B, N, 1, 1) * (1, N, C, rank) -> Sum along N -> (B, C, rank)
        w_view = weights.view(B, self.num_components, 1, 1)
        u_mixed = (w_view * self.u_bank.unsqueeze(0)).sum(dim=1)
        v_mixed = (w_view * self.v_bank.unsqueeze(0)).sum(dim=1)
        
        # 6. Generate M0 via low-rank approximation (M0 = U @ V^T)
        out = torch.bmm(u_mixed, v_mixed.transpose(1, 2))  # (B, C, C)
        
        return out


class EGLAVarNet(nn.Module):
    """Enhanced Gated Linear Attention Variational Network (EGLAVarNet).
    
    Features:
    - Gated Linear Attention (GLA) for global spatial context.
    - Input-Adaptive Prompt Generator for memory initialization.
    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        in_channels: int = COMPLEX_SIZE,
        num_steps: int = 15,
        recurrent_hidden_channels: int = 64,
        recurrent_num_layers: int = 4,
        no_parameter_sharing: bool = True,
        learned_initializer: bool = False,
        initializer_initialization: Optional[InitType] = None,
        initializer_channels: Optional[Tuple[int, ...]] = (32, 32, 64, 64),
        initializer_dilations: Optional[Tuple[int, ...]] = (1, 1, 2, 4),
        initializer_multiscale: int = 1,
        normalized: bool = False,
        **kwargs,
    ):
        super().__init__()

        extra_keys = kwargs.keys()
        for extra_key in extra_keys:
            if extra_key not in [
                "model_name",
            ]:
                raise ValueError(f"{type(self).__name__} got key `{extra_key}` which is not supported.")

        self.initializer: Optional[nn.Module] = None
        if (
            learned_initializer
            and initializer_initialization is not None
        ):
            if initializer_initialization not in [
                "sense",
                "input_image",
                "zero_filled",
            ]:
                raise ValueError(
                    f"Unknown initializer_initialization. Expected `sense`, `'input_image` or `zero_filled`."
                    f"Got {initializer_initialization}."
                )
            self.initializer_initialization = initializer_initialization
            
            # --- EGLAVarNet Change: Use PromptGenerator instead of RecurrentInit ---
            self.initializer = PromptGenerator(
                in_channels=in_channels,
                hidden_channels=recurrent_hidden_channels
            )
            # -----------------------------------------------------------------------

        self.num_steps = num_steps
        self.no_parameter_sharing = no_parameter_sharing
        self.block_list = nn.ModuleList()
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.block_list.append(
                EGLAVarNetBlock(
                    forward_operator=forward_operator,
                    backward_operator=backward_operator,
                    in_channels=in_channels,
                    hidden_channels=recurrent_hidden_channels,
                    num_layers=recurrent_num_layers,
                    normalized=normalized,
                )
            )
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self._coil_dim = 1
        self._spatial_dims = (2, 3)

    def compute_sense_init(self, kspace: torch.Tensor, sensitivity_map: torch.Tensor) -> torch.Tensor:
        r"""Computes sense initialization :math:`x_{\text{SENSE}}`:

        .. math::
            x_{\text{SENSE}} = \sum_{k=1}^{n_c} {S^{k}}^* \times y^k

        where :math:`y^k` denotes the data from coil :math:`k`.

        Parameters
        ----------
        kspace: torch.Tensor
            k-space of shape (N, coil, height, width, complex=2).
        sensitivity_map: torch.Tensor
            Sensitivity map of shape (N, coil, height, width, complex=2).

        Returns
        -------
        input_image: torch.Tensor
            Sense initialization :math:`x_{\text{SENSE}}`.
        """
        input_image = complex_multiplication(
            conjugate(sensitivity_map),
            self.backward_operator(kspace, dim=self._spatial_dims),
        )
        input_image = input_image.sum(self._coil_dim)

        return input_image

    def forward(
        self,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes forward pass of :class:`EGLAVarNet`.

        Parameters
        ----------
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).

        Returns
        -------
        kspace_prediction: torch.Tensor
            k-space prediction.
        """

        previous_state: Optional[torch.Tensor] = None

        if self.initializer is not None:
            if self.initializer_initialization == "sense":
                initializer_input_image = self.compute_sense_init(
                    kspace=masked_kspace,
                    sensitivity_map=sensitivity_map,
                ) # (N, H, W, 2) 
                initializer_input_image = initializer_input_image.permute(0, 3, 1, 2) # To (N, 2, H, W)
                
            elif self.initializer_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initializer_initialization "
                        f"is {self.initializer_initialization}."
                    )
                initializer_input_image = kwargs["initial_image"] # Assuming (N, H, W, 2)
                initializer_input_image = initializer_input_image.permute(0, 3, 1, 2)

            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims)
                initializer_input_image = initializer_input_image.sum(self._coil_dim) # RSS or Sum?
                initializer_input_image = initializer_input_image.permute(0, 3, 1, 2)

            # --- EGLAVarNet Change: Generate Adaptive Memory ---
            # Input: (N, 2, H, W)
            # Output: (N, C, C) - This is our Prompt-initialized Memory
            previous_state = self.initializer(initializer_input_image)
            
            # Replicate memory across recurrent layers (B, L, C, C)
            num_layers_recurrent = self.block_list[0].regularizer.num_layers
            previous_state = previous_state.unsqueeze(1).repeat(1, num_layers_recurrent, 1, 1) # (B, L, C, C)
            # ---------------------------------------------------

        kspace_prediction = masked_kspace.clone()

        for step in range(self.num_steps):
            block = self.block_list[step] if self.no_parameter_sharing else self.block_list[0]
            if self.training:
                kspace_prediction, previous_state = checkpoint(
                    block,
                    kspace_prediction,
                    masked_kspace,
                    sampling_mask,
                    sensitivity_map,
                    previous_state,
                    self._coil_dim,
                    self._spatial_dims,
                    use_reentrant=False
                )
            else:
                kspace_prediction, previous_state = block(
                    kspace_prediction,
                    masked_kspace,
                    sampling_mask,
                    sensitivity_map,
                    previous_state,
                    self._coil_dim,
                    self._spatial_dims,
                )

        return kspace_prediction


class GatedLinearAttention(nn.Module):
    """Gated Linear Attention (GLA) module with Recurrent Form (O(1) inference memory)."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.scale = hidden_channels**-0.5

        # Normalization for stability
        self.norm = nn.GroupNorm(num_groups=8, num_channels=hidden_channels)

        # Projections for Q, K, V
        self.to_q = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.to_k = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.to_v = nn.Conv2d(hidden_channels, hidden_channels, 1)

        # Gating mechanisms
        self.to_gates = nn.Conv2d(hidden_channels, hidden_channels * 2, 1)

        # Final projection
        self.proj = nn.Conv2d(hidden_channels, hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        prev_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        N = H * W

        # 0. Normalization
        x_norm = self.norm(x)

        # 1. Project inputs to Q, K, V, and Gates
        q = self.to_q(x_norm).view(B, C, N) * self.scale
        k = self.to_k(x_norm).view(B, C, N)
        v = self.to_v(x_norm).view(B, C, N)

        gates = self.to_gates(x_norm).view(B, 2 * C, N)
        forget_gate_spatial, input_gate_spatial = gates.chunk(2, dim=1)
        forget_gate_spatial = torch.sigmoid(forget_gate_spatial)
        input_gate_spatial = torch.sigmoid(input_gate_spatial)

        # 2. Compute New Knowledge (Aggregation)
        k_gated = k * input_gate_spatial
        # Normalize by N to prevent memory explosion with large images
        new_knowledge = torch.bmm(k_gated, v.transpose(1, 2)) / N

        # 3. Update Global Memory
        global_forget_gate = forget_gate_spatial.mean(dim=2).unsqueeze(2)

        if prev_memory is None:
            prev_memory = torch.zeros(
                (B, C, C), dtype=x.dtype, device=x.device
            )

        # M_t = lambda * M_{t-1} + NewKnowledge
        new_memory = global_forget_gate * prev_memory + new_knowledge

        # 4. Retrieval (Personalized Query)
        q_rows = q.transpose(1, 2)
        retrieved = torch.bmm(q_rows, new_memory)
        retrieved = retrieved.transpose(1, 2).view(B, C, H, W)

        # 5. Fusion
        out = self.proj(retrieved) + x

        return out, new_memory


class GLARecurrentUnit(nn.Module):
    """Wrapper for Gated Linear Attention to act as a Recurrent Unit (replacement for Conv2dGRU)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # Lifting projection: Input (2) -> Hidden (C)
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)

        # GLA Layers
        self.layers = nn.ModuleList([
            GatedLinearAttention(hidden_channels) for _ in range(num_layers)
        ])

        # Output projection: Hidden (C) -> Output (2)
        self.output_proj = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lift to hidden dimension
        feat = self.input_proj(x)

        new_states_list = []
        if hidden_states is None:
            current_states = [None] * self.num_layers
        else:
            current_states = hidden_states.unbind(dim=1)

        for i, layer in enumerate(self.layers):
            prev_mem = current_states[i]
            feat, new_mem = layer(feat, prev_mem)
            new_states_list.append(new_mem)

        new_hidden_states = torch.stack(new_states_list, dim=1)

        # Project back to input dimension and add residual
        out = self.output_proj(feat) + x

        return out, new_hidden_states


class EGLAVarNetBlock(nn.Module):
    r"""Enhanced Gated Linear Attention Variational Network Block."""

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
        normalized: bool = False,
    ):
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.learning_rate = nn.Parameter(torch.tensor([1.0]))
        regularizer_params = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
        }
        self.regularizer = GLARecurrentUnit(**regularizer_params)

    def forward(
        self,
        current_kspace: torch.Tensor,
        masked_kspace: torch.Tensor,
        sampling_mask: torch.Tensor,
        sensitivity_map: torch.Tensor,
        hidden_state: Union[None, torch.Tensor],
        coil_dim: int = 1,
        spatial_dims: Tuple[int, int] = (2, 3),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kspace_error = torch.where(
            sampling_mask == 0,
            torch.tensor([0.0], dtype=masked_kspace.dtype).to(masked_kspace.device),
            current_kspace - masked_kspace,
        )

        recurrent_term = reduce_operator(
            self.backward_operator(current_kspace, dim=spatial_dims),
            sensitivity_map,
            dim=coil_dim,
        ).permute(0, 3, 1, 2)

        recurrent_term, hidden_state = self.regularizer(recurrent_term, hidden_state)
        recurrent_term = recurrent_term.permute(0, 2, 3, 1)

        recurrent_term = self.forward_operator(
            expand_operator(recurrent_term, sensitivity_map, dim=coil_dim),
            dim=spatial_dims,
        )

        new_kspace = current_kspace - self.learning_rate * kspace_error + recurrent_term

        return new_kspace, hidden_state