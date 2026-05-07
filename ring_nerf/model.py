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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

from nerfstudio.data.scene_box import SceneBox
import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel

from ring_nerf.field import RINGField
from nerfstudio.viewer.viewer_elements import ViewerSlider


@dataclass
class RINGModelConfig(NerfactoModelConfig):
    """RING-NeRF Model Config"""

    _target: Type = field(default_factory=lambda: RINGModel)
    distance_aware_forward: bool = True
    coarse_to_fine: bool = False
    few_shot_loss:bool = False
    few_shot_loss_back:bool = False

class RINGModel(NerfactoModel):    
    
    config: RINGModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super(NerfactoModel, self).populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        if self.config.coarse_to_fine:
            self.coarse_to_fine = self.config.coarse_to_fine
            self.continuous_coarse_to_fine = True
            self.ctf_lvl_init = 4
            self.ctf_epochs = 500
        

        # Fields
        self.field = RINGField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
            distance_aware_forward=self.config.distance_aware_forward,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.lod_slider = ViewerSlider(
            name="LOD",
            default_value=self.config.num_levels - 1,
            min_value=0,
            max_value=self.config.num_levels - 1,
            step=1,
            disabled=False,
            visible=True,
            cb_hook=None,
            hint="Level of Detail for rendering",
        )




    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        if self.training is False:
            lod_display = int(self.lod_slider.value) if self.lod_slider is not None and self.lod_slider.value != self.config.num_levels - 1 else int(self.config.num_levels)
        else:
            lod_display = None
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals, lod_display=lod_display)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        lod = torch.sum(weights * field_outputs["LOD"], dim=-2)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "lod": lod,
        }
        if self.config.few_shot_loss or self.config.few_shot_loss_back and self.training:
            outputs["density"] = field_outputs[FieldHeadNames.DENSITY]

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs




    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
        ) -> List[TrainingCallback]:
            callbacks = super(NerfactoModel,self).get_training_callbacks(training_callback_attributes)
            print("Using RING-NeRF training callbacks")
            if self.config.coarse_to_fine :
                def set_ctf(step):
                    ctf = max(self.ctf_lvl_init,int(step/self.ctf_epochs)) if step < self.ctf_epochs*self.field.num_levels else self.field.num_levels
                    c = ((step % self.ctf_epochs) / self.ctf_epochs if step < self.ctf_epochs*self.field.num_levels else 1) if self.ctf_lvl_init < ctf else 1
                    self.field.mlp_base.coarse_to_fine = ctf
                    self.field.mlp_base.continuous_ctf = c  if self.continuous_coarse_to_fine else 0 

                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=set_ctf,
                    )
                )
            
            return callbacks


    def get_loss_dict(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor],metrics_dict=None) -> Dict[str, Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch,metrics_dict)
        if self.config.few_shot_loss and self.training:
                M= 12
                # print(outputs.keys())
                loss_dict["free_nerf"] = 0.01 * (outputs["density"][:,:M]).mean()
                bw_prior = False ## Depends on the dataset, e.g. DTU which have white & black backgrounds.
                if bw_prior :
                    image = batch["image"].to(self.device)
                    M= 20 
                    rgb = image.mean(dim=-1)
                    mask_w = torch.where(rgb > 0.99,1,0)
                    mask_b = torch.where(rgb < 0.01,1,0)
                    mask = mask_w + mask_b
                    mask = mask.unsqueeze(-1).expand(len(mask),outputs["density"].shape[1])
                    mask[:,M:] = 0
                    loss_dict["free_nerf_bw"] = 0.01 * (outputs["density"]*mask).mean()
        if self.config.few_shot_loss_back != 0 and self.training:
            M= 12
            loss_dict["free_nerf"] = 0.01 * (outputs["density"][:,-M:]).mean()
        return loss_dict
