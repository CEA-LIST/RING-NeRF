# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Multi Layer Perceptron
"""
from typing import Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.utils.external import TCNN_EXISTS, tcnn
from nerfstudio.utils.printing import print_tcnn_speed_warning
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.field_components.mlp import MLP


class RINGNetwork(FieldComponent):
    """Multilayer perceptron with hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
        num_layers: int = 2,
        layer_width: int = 64,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()
        self.in_dim = 3

        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.features_per_level = features_per_level
        self.hash_init_scale = hash_init_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1

        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        resolutions = self.min_res * (self.max_res / self.min_res) ** (torch.arange(num_levels) / (num_levels - 1))        
        self.register_buffer("resolutions", resolutions)

        self.tcnn_encoding = tcnn.Encoding(
            n_input_dims=self.in_dim,
            encoding_config=HashEncoding.get_tcnn_encoding_config(
                num_levels=self.num_levels,
                features_per_level=self.features_per_level,
                log2_hashmap_size=self.log2_hashmap_size,
                min_res=self.min_res,
                growth_factor=self.growth_factor,
                interpolation=interpolation,
            )
        )

        # self.tcnn_network = tcnn.Network(
        #     n_input_dims=self.tcnn_encoding.n_output_dims,
        #     n_output_dims=self.out_dim,
        #     network_config=MLP.get_tcnn_network_config(
        #         activation=self.activation,
        #         out_activation=self.out_activation,
        #         layer_width=self.layer_width,
        #         num_layers=self.num_layers,
        #     ),
        # )

        ## Initialization Changes. Do not change much to the results, but supposed to be cleaner, especially for coarse-to-fine and resolution extensibility.
        first_par =[(self.min_res*2**k)**3 for k in range(self.num_levels) if (k+int(np.log2(self.min_res)))*3 < self.log2_hashmap_size]
        par = first_par + [2**self.log2_hashmap_size for k in range(self.num_levels - len(first_par))]
        nb_params = self.features_per_level*(sum(par[:1]))
        nb_params=self.features_per_level*(self.min_res**3)
        with torch.no_grad():
            self.tcnn_encoding.params[:nb_params] = torch.nn.Parameter(torch.rand(self.tcnn_encoding.params[:nb_params].shape)).to(self.tcnn_encoding.params.device)
            self.tcnn_encoding.params[nb_params:] = torch.nn.Parameter(torch.zeros_like(self.tcnn_encoding.params[nb_params:])).to(self.tcnn_encoding.params.device)
        
        self.layer_norm = torch.nn.LayerNorm(self.features_per_level, elementwise_affine=False)
        self.sin_layer = torch.nn.Linear(self.features_per_level, self.layer_width, bias=False)
        self.net = torch.nn.Linear(self.layer_width, self.out_dim)

        self.coarse_to_fine = self.num_levels
        self.continuous_ctf = 1

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"], lod=None,lod_display=None,include_intermediate_steps=False) -> Float[Tensor, "*bs out_dim"]:
        encoding = self.tcnn_encoding(in_tensor)
        
        if lod_display is None :
            lod_display = int(self.num_levels)

        if lod is not None:
            lod_weights = lod - torch.arange(self.num_levels, device=in_tensor.device) + 1
            lod_weights = lod_weights.clamp(0, 1)
            lod_weights = lod_weights.permute(1,0).unsqueeze(-1)
            lod_weights[self.coarse_to_fine-1] *= self.continuous_ctf



        encoding = torch.cat(torch.split(encoding.unsqueeze(0), [self.features_per_level]*self.num_levels, dim=-1))[:self.coarse_to_fine]
        if include_intermediate_steps:
            x = torch.cumsum(encoding, dim=0)
        else:
            if len(lod_weights) == 0 :
                x = torch.sum(encoding, dim=0)
            else :
                # print(lod_weights.shape, encoding.shape,lod_display)
                x = (lod_weights[:len(encoding)]*encoding)[:lod_display].sum(dim=0)


        x = self.layer_norm(x)
        z = torch.sin(self.sin_layer(x.float()))
        
        return self.net(z)
