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

from direct.constants import COMPLEX_SIZE
from direct.data.transforms import complex_multiplication, conjugate, expand_operator, reduce_operator
from direct.nn.recurrent.recurrent import Conv2dGRU, NormConv2dGRU
from direct.nn.types import InitType


class RecurrentInit(nn.Module):
    """Recurrent State Initializer (RSI) module of Recurrent Variational Network as presented in [1]_.

    The RSI module learns to initialize the recurrent hidden state :math:`h_0`, input of the first RecurrentVarNetBlock
    of the RecurrentVarNet.

    References
    ----------

    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
        the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...],
        dilations: Tuple[int, ...],
        depth: int = 2,
        multiscale_depth: int = 1,
    ):
        """Inits :class:`RecurrentInit`.

        Parameters
        ----------
        in_channels: int
            Input channels.
        out_channels: int
            Number of hidden channels of the recurrent unit of RecurrentVarNet Block.
        channels: tuple
            Channels :math:`n_d` in the convolutional layers of initializer.
        dilations: tuple
            Dilations :math:`p` of the convolutional layers of the initializer.
        depth: int
            RecurrentVarNet Block number of layers :math:`n_l`.
        multiscale_depth: 1
            Number of feature layers to aggregate for the output, if 1, multi-scale context aggregation is disabled.
        """
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        self.depth = depth
        self.multiscale_depth = multiscale_depth
        tch = in_channels
        for curr_channels, curr_dilations in zip(channels, dilations):
            block = [
                nn.ReplicationPad2d(curr_dilations),
                nn.Conv2d(tch, curr_channels, 3, padding=0, dilation=curr_dilations),
            ]
            tch = curr_channels
            self.conv_blocks.append(nn.Sequential(*block))
        tch = np.sum(channels[-multiscale_depth:])
        for _ in range(depth):
            block = [nn.Conv2d(tch, out_channels, 1, padding=0)]
            self.out_blocks.append(nn.Sequential(*block))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes initialization for recurrent unit given input `x`.

        Parameters
        ----------
        x: torch.Tensor
            Initialization for RecurrentInit.

        Returns
        -------
        out: torch.Tensor
            Initial recurrent hidden state from input `x`.
        """

        features = []
        for block in self.conv_blocks:
            x = F.relu(block(x), inplace=True)
            if self.multiscale_depth > 1:
                features.append(x)
        if self.multiscale_depth > 1:
            x = torch.cat(features[-self.multiscale_depth :], dim=1)
        output_list = []
        for block in self.out_blocks:
            y = F.relu(block(x), inplace=True)
            output_list.append(y)
        out = torch.stack(output_list, dim=-1)
        return out


class RecurrentVarNet(nn.Module):
    """Recurrent Variational Network implementation as presented in [1]_.

    References
    ----------

    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
        the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.
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
        """Inits :class:`RecurrentVarNet`.

        Parameters
        ----------
        forward_operator: Callable
            Forward Operator.
        backward_operator: Callable
            Backward Operator.
        num_steps: int
            Number of iterations :math:`T`.
        in_channels: int
            Input channel number. Default is 2 for complex data.
        recurrent_hidden_channels: int
            Hidden channels number for the recurrent unit of the RecurrentVarNet Blocks. Default: 64.
        recurrent_num_layers: int
            Number of layers for the recurrent unit of the RecurrentVarNet Block (:math:`n_l`). Default: 4.
        no_parameter_sharing: bool
            If False, the same :class:`RecurrentVarNetBlock` is used for all num_steps. Default: True.
        learned_initializer: bool
            If True an RSI module is used. Default: False.
        initializer_initialization: str, Optional
            Type of initialization for the RSI module. Can be either 'sense', 'zero-filled' or 'input-image'.
            Default: None.
        initializer_channels: tuple
            Channels :math:`n_d` in the convolutional layers of the RSI module. Default: (32, 32, 64, 64).
        initializer_dilations: tuple
            Dilations :math:`p` of the convolutional layers of the RSI module. Default: (1, 1, 2, 4).
        initializer_multiscale: int
            RSI module number of feature layers to aggregate for the output, if 1, multi-scale context aggregation
            is disabled. Default: 1.
        normalized: bool
            If True, :class:`NormConv2dGRU` will be used as a regularizer in the :class:`RecurrentVarNetBlocks`.
            Default: False.
        """
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
            and initializer_channels is not None
            and initializer_dilations is not None
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
            self.initializer = RecurrentInit(
                in_channels,
                recurrent_hidden_channels,
                channels=initializer_channels,
                dilations=initializer_dilations,
                depth=recurrent_num_layers,
                multiscale_depth=initializer_multiscale,
            )
        self.num_steps = num_steps
        self.no_parameter_sharing = no_parameter_sharing
        self.block_list = nn.ModuleList()
        for _ in range(self.num_steps if self.no_parameter_sharing else 1):
            self.block_list.append(
                RecurrentVarNetBlock(
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
        """Computes forward pass of :class:`RecurrentVarNet`.

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
                ).unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "input_image":
                if "initial_image" not in kwargs:
                    raise ValueError(
                        f"`'initial_image` is required as input if initializer_initialization "
                        f"is {self.initializer_initialization}."
                    )
                initializer_input_image = kwargs["initial_image"].unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = self.backward_operator(masked_kspace, dim=self._spatial_dims)

            # previous_state = self.initializer(
            #     self.forward_operator(initializer_input_image, dim=self._spatial_dims)
            #     .sum(self._coil_dim)
            #     .permute(0, 3, 1, 2)
            # )
            # GLA Adaptation: The RecurrentInit module outputs spatial hidden states for Conv2dGRU.
            # GLA expects a global memory matrix [B, L, C, C].
            # For now, we disable the explicit initialization from RecurrentInit and start with zero memory.
            previous_state = None

        kspace_prediction = masked_kspace.clone()

        for step in range(self.num_steps):
            block = self.block_list[step] if self.no_parameter_sharing else self.block_list[0]
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
    """Gated Linear Attention (GLA) module with Recurrent Form (O(1) inference memory).

    This module implements a global attention mechanism that maintains a fixed-size memory matrix
    updated at each iteration, rather than storing all historical feature maps.

    References
    ----------
    Inspired by Linear Attention and Gated Linear Networks theories.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.scale = hidden_channels**-0.5

        # Projections for Q, K, V
        self.to_q = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.to_k = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.to_v = nn.Conv2d(hidden_channels, hidden_channels, 1)

        # Gating mechanisms
        # lambda_gate: Forget gate (0~1)
        # input_gate: Input importance gate (0~1)
        self.to_gates = nn.Conv2d(hidden_channels, hidden_channels * 2, 1)

        # Final projection
        self.proj = nn.Conv2d(hidden_channels, hidden_channels, 1)

    def forward(
        self, x: torch.Tensor, prev_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C, H, W).
        prev_memory : torch.Tensor, optional
            Previous global memory matrix of shape (B, C, C).

        Returns
        -------
        out : torch.Tensor
            Output feature map of shape (B, C, H, W).
        new_memory : torch.Tensor
            Updated global memory matrix of shape (B, C, C).
        """
        B, C, H, W = x.shape
        N = H * W

        # 1. Project inputs to Q, K, V, and Gates
        q = self.to_q(x).view(B, C, N)  # (B, C, N)
        k = self.to_k(x).view(B, C, N)  # (B, C, N)
        v = self.to_v(x).view(B, C, N)  # (B, C, N)

        gates = self.to_gates(x).view(B, 2 * C, N)
        forget_gate_spatial, input_gate_spatial = gates.chunk(2, dim=1)
        forget_gate_spatial = torch.sigmoid(forget_gate_spatial)  # lambda_t (B, C, N)
        input_gate_spatial = torch.sigmoid(input_gate_spatial)  # i_t (B, C, N)

        # 2. Compute New Knowledge (Aggregation)
        # We want to compute: sum_{spatial} (input_gate * K)^T * V
        # (B, C, N) -> (B, N, C) @ (B, N, C) is expensive if we transpose N.
        # Let's optimize: (input_gate * k) @ v.T
        # Shape: (B, C, N) @ (B, N, C) -> (B, C, C)
        # Note: (i * K) is element-wise.
        k_gated = k * input_gate_spatial  # (B, C, N)
        new_knowledge = torch.bmm(k_gated, v.transpose(1, 2))  # (B, C, C)

        # 3. Update Global Memory
        # Compute global forgetting factor.
        # We can take the mean of spatial forget gates to get a global forget gate for the matrix.
        # global_forget_gate = forget_gate_spatial.mean(dim=2).unsqueeze(2)  # (B, C, 1) -> broadcasting to (B, C, C)
        # Alternatively, simpler element-wise decay if prev_memory is (B, C, C).
        # Let's use a learned global scalar/vector derived from spatial map for stability.
        global_forget_gate = forget_gate_spatial.mean(dim=2).unsqueeze(2)  # (B, C, 1)

        if prev_memory is None:
            prev_memory = torch.zeros(
                (B, C, C), dtype=x.dtype, device=x.device
            )

        # M_t = lambda * M_{t-1} + NewKnowledge
        new_memory = global_forget_gate * prev_memory + new_knowledge

        # 4. Retrieval (Personalized Query)
        # Output = Q @ M_t
        # Shape: (B, N, C) @ (B, C, C) -> (B, N, C) -> transpose back to (B, C, N)
        # Wait, Q is (B, C, N). So we want Q.T @ M_t ?
        # Query vector q_i is (1, C). M is (C, C). Result is (1, C).
        # So we want result of shape (B, C, N).
        # result_column = M.T @ q_column ?
        # Let's look at algebra: y = q^T M. (where q is column vector C x 1).
        # If Q is (B, C, N), then we treat it as N column vectors.
        # result = M_t.transpose(1, 2) @ Q ? No.
        # Let's stick to the user's formula: Output = Q_point . M
        # If Q_point is row vector (1, C): result is (1, C).
        # Here Q is (B, C, N), so it's N column vectors.
        # Let's view Q as (B, N, C) (batch of N row vectors).
        q_rows = q.transpose(1, 2)  # (B, N, C)
        retrieved = torch.bmm(q_rows, new_memory)  # (B, N, C) @ (B, C, C) -> (B, N, C)
        retrieved = retrieved.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

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
        self, x: torch.Tensor, hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input (recurrent term) of shape (B, in_channels, H, W).
        hidden_states : torch.Tensor, optional
            Stacked memory matrices of shape (B, num_layers, C, C).

        Returns
        -------
        out : torch.Tensor
            Updated input of shape (B, in_channels, H, W).
        new_hidden_states : torch.Tensor
            Updated memory matrices of shape (B, num_layers, C, C).
        """
        # Lift to hidden dimension
        feat = self.input_proj(x)  # (B, C, H, W)

        new_states_list = []
        if hidden_states is None:
            # Initialize with None
            current_states = [None] * self.num_layers
        else:
            # Unbind the stacked states: (B, L, C, C) -> List of (B, C, C)
            current_states = hidden_states.unbind(dim=1)

        for i, layer in enumerate(self.layers):
            prev_mem = current_states[i]
            feat, new_mem = layer(feat, prev_mem)
            new_states_list.append(new_mem)

        # Stack new states
        new_hidden_states = torch.stack(new_states_list, dim=1)  # (B, L, C, C)

        # Project back to input dimension and add residual
        out = self.output_proj(feat) + x

        return out, new_hidden_states


class RecurrentVarNetBlock(nn.Module):
    r"""Recurrent Variational Network Block :math:`\mathcal{H}_{\theta_{t}}` as presented in [1]_.

    References
    ----------

    .. [1] Yiasemis, George, et al. “Recurrent Variational Network: A Deep Learning Inverse Problem Solver Applied to
        the Task of Accelerated MRI Reconstruction.” ArXiv:2111.09639 [Physics], Nov. 2021. arXiv.org,
        http://arxiv.org/abs/2111.09639.

    """

    def __init__(
        self,
        forward_operator: Callable,
        backward_operator: Callable,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
        normalized: bool = False,
    ):
        """Inits RecurrentVarNetBlock.

        Parameters
        ----------
        forward_operator: Callable
            Forward Fourier Transform.
        backward_operator: Callable
            Backward Fourier Transform.
        in_channels: int,
            Input channel number. Default is 2 for complex data.
        hidden_channels: int,
            Hidden channels. Default: 64.
        num_layers: int,
            Number of layers of :math:`n_l` recurrent unit. Default: 4.
        normalized: bool
            If True, :class:`NormConv2dGRU` will be used as a regularizer. Default: False.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator

        self.learning_rate = nn.Parameter(torch.tensor([1.0]))  # :math:`\alpha_t`
        regularizer_params = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            # "replication_padding": True, # GLA uses standard padding in projections
        }
        # Replaced Conv2dGRU with GLARecurrentUnit
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
        """Computes forward pass of RecurrentVarNetBlock.

        Parameters
        ----------
        current_kspace: torch.Tensor
            Current k-space prediction of shape (N, coil, height, width, complex=2).
        masked_kspace: torch.Tensor
            Masked k-space of shape (N, coil, height, width, complex=2).
        sampling_mask: torch.Tensor
            Sampling mask of shape (N, 1, height, width, 1).
        sensitivity_map: torch.Tensor
            Coil sensitivities of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor or None
            Recurrent unit hidden state of shape (N, L, C, C) if not None. Optional.
        coil_dim: int
            Coil dimension. Default: 1.
        spatial_dims: tuple of ints
            Spatial dimensions. Default: (2, 3).

        Returns
        -------
        new_kspace: torch.Tensor
            New k-space prediction of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor
            Next hidden state of shape (N, L, C, C).
        """

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

        recurrent_term, hidden_state = self.regularizer(recurrent_term, hidden_state)  # :math:`w_t`, :math:`h_{t+1}`
        recurrent_term = recurrent_term.permute(0, 2, 3, 1)

        recurrent_term = self.forward_operator(
            expand_operator(recurrent_term, sensitivity_map, dim=coil_dim),
            dim=spatial_dims,
        )

        new_kspace = current_kspace - self.learning_rate * kspace_error + recurrent_term

        return new_kspace, hidden_state  # type: ignore
