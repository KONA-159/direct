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


class RecurrentVarNet3D(nn.Module):
    """Recurrent Variational Network implementation adapted for 3D (2D+t) data.
    
    This class wraps the 2D RecurrentVarNet logic but handles 3D input tensors (N, C, T, H, W, 2)
    by folding the time dimension T into the batch dimension N.
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
        """Computes forward pass of :class:`RecurrentVarNet3D`."""
        
        # Shapes: masked_kspace is (N, Coil, Time, H, W, 2)
        N, C, T, H, W, Complex = masked_kspace.shape
        
        # Reshape to (N*T, C, H, W, Complex) to use 2D components
        masked_kspace_2d = masked_kspace.permute(0, 2, 1, 3, 4, 5).reshape(N * T, C, H, W, Complex)
        sensitivity_map_2d = sensitivity_map.permute(0, 2, 1, 3, 4, 5).reshape(N * T, C, H, W, Complex)
        
        # Handle sampling_mask broadcasting if T dimension is 1
        if sampling_mask.shape[2] == 1 and T > 1:
            sampling_mask = sampling_mask.expand(-1, -1, T, -1, -1, -1)
            
        sampling_mask_2d = sampling_mask.permute(0, 2, 1, 3, 4, 5).reshape(N * T, 1, H, W, 1)

        previous_state: Optional[torch.Tensor] = None

        if self.initializer is not None:
            if self.initializer_initialization == "sense":
                initializer_input_image = self.compute_sense_init(
                    kspace=masked_kspace_2d,
                    sensitivity_map=sensitivity_map_2d,
                ).unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "input_image":
                initializer_input_image = kwargs["initial_image"].reshape(N * T, H, W, Complex).unsqueeze(self._coil_dim)
            elif self.initializer_initialization == "zero_filled":
                initializer_input_image = self.backward_operator(masked_kspace_2d, dim=self._spatial_dims)

            previous_state = self.initializer(
                self.forward_operator(initializer_input_image, dim=self._spatial_dims)
                .sum(self._coil_dim)
                .permute(0, 3, 1, 2)
            )

        kspace_prediction_2d = masked_kspace_2d.clone()

        for step in range(self.num_steps):
            block = self.block_list[step] if self.no_parameter_sharing else self.block_list[0]
            kspace_prediction_2d, previous_state = block(
                kspace_prediction_2d,
                masked_kspace_2d,
                sampling_mask_2d,
                sensitivity_map_2d,
                previous_state,
                self._coil_dim,
                self._spatial_dims,
            )
            
        # Unfold Time from Batch: (N*T, C, H, W, 2) -> (N, T, C, H, W, 2) -> (N, C, T, H, W, 2)
        kspace_prediction = kspace_prediction_2d.view(N, T, C, H, W, Complex).permute(0, 2, 1, 3, 4, 5)

        return kspace_prediction


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
            "replication_padding": True,
        }
        # Recurrent Unit of RecurrentVarNet Block :math:`\mathcal{H}_{\theta_t}`
        self.regularizer = (
            NormConv2dGRU(**regularizer_params) if normalized else Conv2dGRU(**regularizer_params)  # type: ignore
        )

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
            Recurrent unit hidden state of shape (N, hidden_channels, height, width, num_layers) if not None. Optional.
        coil_dim: int
            Coil dimension. Default: 1.
        spatial_dims: tuple of ints
            Spatial dimensions. Default: (2, 3).

        Returns
        -------
        new_kspace: torch.Tensor
            New k-space prediction of shape (N, coil, height, width, complex=2).
        hidden_state: torch.Tensor
            Next hidden state of shape (N, hidden_channels, height, width, num_layers).
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
