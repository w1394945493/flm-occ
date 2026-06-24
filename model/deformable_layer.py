# Adopt from Horizon Robotics
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.math import  rotation_from_quaternion, safe_sigmoid, normalize_intrinsics
from utils.init import constant_init, xavier_init
from model import scene



def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class DeformableFeatureAggregation(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kpts_generator = None,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        dim_factor=2,
        out_of_frustum=False,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups) # 32
        self.embed_dims = embed_dims # 96
        self.num_levels = num_levels # 3
        self.num_groups = num_groups # 3
        self.num_cams = num_cams # 1
        # self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop # 0.15
        self.residual_mode = residual_mode # "add"
        self.proj_drop = nn.Dropout(proj_drop)
        self.kps_generator = kpts_generator()
        self.num_pts = self.kps_generator.num_pts
        # self.proj_in is not used in the original deform_aggEGO
        # self.proj_in = nn.Linear(int(embed_dims/dim_factor), embed_dims) if dim_factor > 1 else nn.Identity()
        self.proj_in = nn.Linear(int(embed_dims/dim_factor), embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.out_of_frustum = out_of_frustum

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 3*2) # only use intrinsics here
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.proj_in, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        BGSfeat: torch.Tensor,
        BGS: torch.Tensor,
        BGSembed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        BK,  # [B,3,3]
        BWH, # [B,2]
    ):
        """
        B: batch size
        M: number of Gaussians
        C: embed_dims
        P: number of keypoints per Gaussian
        V: number of cameras
        L: number of feature levels
        G: number of groups
        """
        B, num_gs = BGSfeat.shape[:2]
        key_points = self.kps_generator(BGS, BGSfeat, BK, BWH) # [B, M, P, 3]
        weights, mask = self._get_weights(
            BGSfeat, BGSembed,
            BK[:, None],      # [B, V, 3, 3], V=1
            BWH[:, None]      # [B, V, 2],    V=1
        )                     # [B, M, P, V, L, G]

        points_2d, mask_2d = self.project_points(
            key_points.flatten(1, 2), # [B, M*P, 3]
            BK[:, None],              # [B, V, 3, 3], V=1
            BWH[:, None],             # [B, V, 2],    V=1
            ret_mask=self.out_of_frustum,
        ) # [B, M*P, V, 2], [B, M*P, V]
        if self.out_of_frustum:
            mask_2d = mask_2d.reshape(B, num_gs, self.num_pts, self.num_cams) # [B, M, P, V]
            mask_2d = mask_2d[..., None, None] # [B, M, P, V, 1, 1]
            mask = mask & mask_2d              # [B, M, P, V, L, G]
            
        weights[~mask] = - torch.inf
        all_drop = mask.sum(dim=[2,3,4], keepdim=True) == 0  # [B, M, 1, 1, 1, 1]
        weights[all_drop.expand_as(weights)] = 0.            # [B, M, P, V, L, G]
        weights = weights.flatten(2, 4).softmax(dim=-2).reshape(
            B,
            num_gs * self.num_pts,
            self.num_cams,
            self.num_levels,
            self.num_groups
        )                                                     # [B, M*P, V, L, G]

        # temp_features_next = DAF.apply(
        #     *feature_maps, points_2d, weights
        # ).reshape(B, num_gs, self.num_pts, self.embed_dims)
        temp_features_next = self._DAF(
            feature_maps, points_2d*2-1, weights
        ).reshape(B, num_gs, self.num_pts, self.embed_dims) # [B, M, P, C]
        features = temp_features_next.sum(dim=-2)  # [B, M, C] fuse multi-point features 
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + BGSfeat
        elif self.residual_mode == "cat":
            output = torch.cat([output, BGSfeat], dim=-1)
        return output # [B, M, C]

    def _DAF(self, feature_maps, points_2d, weights):
        """Replace CUDA implementation of deformable aggregation with PyTorch implementation
        Note that this impl does not support multi-view input.
        
        
        This impl seems slightly faster than the CUDA one.
        
        Note:
            V=1 for monocular cases
            L=4
            G=4
        Args:
            feature_maps (list[Tensor]): [[B, V, C, H, W], ...], assume V=1 camera
            points_2d (Tensor): [B, M*P, V, 2]
            weights (Tensor): [B, M*P, V, L, G]
        """
        B = feature_maps[0].shape[0]
        
        weights = weights.permute(0, 1, 3, 4, 2).contiguous() # [B, M*P, L, G, V]
        
        list_features = []
        for i in range(len(feature_maps)):
            features = F.grid_sample(feature_maps[i][:,0], points_2d, align_corners=True)  # [B, C, M*P, V]
            list_features.append(features)
        features = torch.cat(list_features, dim=-1)          # [B, C, M*P, L]
        features = features.permute(0, 2, 3, 1).contiguous() # [B, M*P, L, C]
        features = self.proj_in(features)                    # [B, M*P, L, C*dim_factor]
        features = features.view(B, -1, self.num_levels, self.num_groups, self.group_dims) # [B, M*P, L, G, C//G]
        feat_agg = (features * weights).sum(dim=-2)          # [B, M*P, L, C//G]
        output = feat_agg.reshape(B, -1, self.embed_dims)    # [B, M*P, C]
        
        return output

    def _get_weights(self, instance_feature, gs_embed, BK, BWH):
        bs, num_gs = instance_feature.shape[:2]
        feature = instance_feature + gs_embed # [B, M, C]
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(normalize_intrinsics(BK, BWH)[...,:2,:].flatten(2,3))
            feature = feature[:, :, None] + camera_embed[:, None]   # [B, M, V, C]
        
        weights = (
            self.weights_fc(feature)                    # [B, M, V, L*P*G]
            # .reshape(bs, num_gs, -1, self.num_groups) # [B, M, V*L*P, G]
            # .softmax(dim=-2)
            .reshape(
                bs,
                num_gs,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            ).permute(0, 1, 4, 2, 3, 5).contiguous()
        ) # [B, M, P, V, L, G]
        if self.training and self.attn_drop > 0:
            # mask = torch.rand(
            #     bs, num_gs, self.num_cams, 1, self.num_pts, 1
            # )
            # mask = mask.to(device=weights.device, dtype=weights.dtype)
            # weights = ((mask > self.attn_drop) * weights) / (
            #     1 - self.attn_drop
            # )
            mask = torch.rand_like(weights)
            mask = mask > self.attn_drop
        else:
            mask = torch.ones_like(weights, dtype=torch.bool)
        return weights, mask

    @staticmethod
    def project_points(key_points, BK, BWH, ret_mask=False):
        points_c = torch.matmul(
            BK.unsqueeze(-3),              # [B, V,   1, 3, 3]
            key_points[:, None, ..., None] # [B, 1, M*P, 3, 1]
        )[..., 0] # [B, V, M*P, 3]
        
        points_2d = points_c[..., :2] / torch.clamp(
            points_c[..., 2:3], min=1e-5
        )
        points_2d = points_2d / BWH.unsqueeze(-2) # [B, V, M*P, 2]
        
        mask = None
        if ret_mask:
            mask = (
                (points_c[..., 2] > 0) 
                & (points_2d[..., 0] >= 0) & (points_2d[..., 0] < 1)
                & (points_2d[..., 1] >= 0) & (points_2d[..., 1] < 1)
            ) # [B, V, M, P]
            mask = mask.permute(0, 2, 1).contiguous()  # [B, M*P, V]

        points_2d = points_2d.permute(0, 2, 1, 3)      # [B, M*P, V, 2]
        
        return points_2d.contiguous(), mask


class SparseGaussian3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
        pc_range=None,
        scale_range=None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.register_buffer('fix_scale', torch.tensor(fix_scale, dtype=torch.float), False)
        self.num_pts = len(self.fix_scale) + num_learnable_pts # 7
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.pc_range = pc_range
        self.scale_range = scale_range

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature,
        BK,
        BWH,
    ):
        
        B, num_anchor = anchor.shape[:2]
        
        fix_scale = self.fix_scale                                    # [P, 3]
        scale = fix_scale[None, None].expand([B, num_anchor, -1, -1]) # [B, M, P, 3]
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                safe_sigmoid(self.learnable_fc(instance_feature)
                .reshape(B, num_anchor, self.num_learnable_pts, 3))
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale], dim=-2)
        
        gs_scales = scene.Superquadrics.scale_transform(anchor[..., None, 3:6]) # [B, G, 1, 3]

        kpts_offsets = scale * gs_scales               # [B, G, P, 3]
        quats = anchor[..., 6:10]
        rotation_mat = rotation_from_quaternion(quats) # [B, G, 3, 3]
        
        kpts_offsets = (rotation_mat.unsqueeze(-3) @ kpts_offsets[..., None]).squeeze(-1) # [B, G, P, 3]
        
        BXYZ = scene.Superquadrics.contracted_to_euclidean(anchor[..., :3], BK, BWH)
        key_points = BXYZ.unsqueeze(2) + kpts_offsets

        return key_points
