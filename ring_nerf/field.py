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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from nerfstudio.fields.nerfacto_field import NerfactoField

from nerfstudio.cameras.rays import Frustums

from torch.func import jacrev

from ring_nerf.mlp import RINGNetwork


class RINGField(NerfactoField):

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        distance_aware_forward: bool = True,
    ) -> None:
        super(NerfactoField, self).__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res
        self.average_init_density = average_init_density
        self.step = 0

        self.distance_aware_forward = distance_aware_forward

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation
        )

        self.mlp_base = RINGNetwork(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
                implementation=implementation,
        )

        # transients
        if self.use_transient_embedding:
            self.transient_embedding_dim = transient_embedding_dim
            self.embedding_transient = Embedding(self.num_images, self.transient_embedding_dim)
            self.mlp_transient = MLP(
                in_dim=self.geo_feat_dim + self.transient_embedding_dim,
                num_layers=num_layers_transient,
                layer_width=hidden_dim_transient,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_transient_uncertainty = UncertaintyFieldHead(in_dim=self.mlp_transient.get_out_dim())
            self.field_head_transient_rgb = TransientRGBFieldHead(in_dim=self.mlp_transient.get_out_dim())
            self.field_head_transient_density = TransientDensityFieldHead(in_dim=self.mlp_transient.get_out_dim())

        # semantics
        if self.use_semantics:
            self.mlp_semantics = MLP(
                in_dim=self.geo_feat_dim,
                num_layers=2,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_semantics = SemanticFieldHead(
                in_dim=self.mlp_semantics.get_out_dim(), num_classes=num_semantic_classes
            )

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = MLP(
                in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.get_out_dim())

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # base values for LOD computation
        self.log_growth_factor = (np.log(max_res) - np.log(base_res)) / (num_levels - 1)
        self.log_base_res = np.log(base_res)

    def distort_positions(self, positions: Tensor) -> Tensor:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
        return positions

    def get_density(self, ray_samples: RaySamples, lod_display: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""

        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)

        if self.distance_aware_forward :  # LOD
            #### For Areat Footprint LOD to be working (which is the one used in the RING-NeRF paper), you need to replace the cameras.py file inside the camera folder in nerfstudio.
            # The modification is really simple (just add the separate ray directions in the metadata), but I didn't find a way to retrieve them differently.
            # I could have added in this repo all the files that import cameras and modify the import to point to the modified cameras.py, but I think it's easier and way lighter to just modify the cameras.py file in place. The change is around line 915, where the ray directions are computed. 
            use_area_footprint = False

            if use_area_footprint:
                pos = positions.detach()
                origins = ray_samples.frustums.origins
                timesteps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
                dir_x = ray_samples.metadata.pop("dir_x")
                dir_y = ray_samples.metadata.pop("dir_y")
                dir_xy = torch.stack([dir_x, dir_y], dim=-2)
                pos_xy = origins.unsqueeze(-2) + timesteps.unsqueeze(-2) * dir_xy
                distorted_xy = self.distort_positions(pos_xy)
                dxy = torch.sqrt(((pos.unsqueeze(-2) - distorted_xy) ** 2).sum(-1))
                log_footprint = torch.log(dxy.prod(-1, keepdim=True)) / 2
            else:
                dist = torch.linalg.norm(ray_samples.frustums.get_positions().detach(), dim=-1)[..., None]
                mag = torch.linalg.norm(ray_samples.frustums.get_positions().detach(), ord=self.spatial_distortion.order, dim=-1)[..., None]
                log_det = torch.where(mag <= 1, torch.zeros_like(mag), torch.log(2 - 1 / mag) * 2 - torch.log(mag) * 4)
                pixel_area = ray_samples.frustums.pixel_area
                log_footprint = torch.log(dist) + torch.log(pixel_area) / 2 + log_det / 3

            lod = -(log_footprint + self.log_base_res) / self.log_growth_factor
            lod = lod.clamp(min=0, max=self.num_levels - 1)
            lod = lod * selector[..., None]
            lod_flat = lod.view(-1, 1)
        else:
            lod = None
            lod_flat = None

        h = self.mlp_base(positions_flat, lod_flat,lod_display=lod_display).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out, lod

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, lod_display: Optional[int] = None) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
            lod_display: Level of detail for rendering.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding, lod = self.get_density(ray_samples, lod_display=lod_display)
        else:
            density, density_embedding, lod = self.get_density(ray_samples, lod_display=lod_display)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        field_outputs["LOD"] = lod

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
