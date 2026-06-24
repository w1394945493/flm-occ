import logging
printi, printd = logging.getLogger('flm-occ').info, logging.getLogger('flm-occ').debug
from functools import partial
import math

import torch
from torch import nn
import torch.nn.functional as F

from model.rgb_encoders import RGBEncoder # DAv2
from model import scene
from model.gaussian_former import SparseGaussianFormer
from model.deformable_layer import DeformableFeatureAggregation, SparseGaussian3DKeyPointsGenerator, linear_relu_ln
from model.FFN import FFN
# from model.refine_layer import SparseGaussian3DDeltaRefinementModule
from model.attention import MHA3D
from model import GMM, SQMM
from configs.base import Model

from utils.math import safe_sigmoid, safe_tanh, safe_inverse_sigmoid



class GaussianLifter(nn.Module):
    def __init__(self,
            num_geometrics=13,
            num_semantics=0,
            dim_feat=128,
        ):
        super().__init__()
        self.instance_feature_layer = nn.Linear(num_geometrics + num_semantics, dim_feat)
    
    def forward(self, GS, feature_maps):
        """
        Use only DPT featmap to initial GSfeat
        """
        B = feature_maps[0].shape[0]
        BGSfeat = self.instance_feature_layer(GS)[None].expand(B, -1, -1)
        BGS = GS[None].expand(B, -1, -1)  # [B, N, 3+3+4+1+2]
        
        return BGS, BGSfeat


class SparseGaussian3DEncoder(nn.Module):
    def __init__(
            self, 
            embed_dims: int = 256, # 96
            num_geometrics: int = 13, # 3 + 3 + 4 + 1 + 2
            num_semantics: int = 0, # 13
        ):
        super().__init__()
        self.embed_dims = embed_dims

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.anchor_dim = num_geometrics + num_semantics
        self.encode_fc = embedding_layer(self.anchor_dim)
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, BGS: torch.Tensor):
        output = self.encode_fc(BGS)
        output = self.output_fc(output)
        return output


class SparseGaussian3DDeltaRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_geometrics=13, # 3 + 3 + 4 + 1 + 2
        num_semantics=0, # 13
        affine_dxyz=lambda x: x * 0.1,
        num_refine=4,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_geometrics = num_geometrics
        self.num_semantics = num_semantics
        output_dim = num_geometrics + num_semantics
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, output_dim),
        )
        self.scale = nn.Parameter(torch.tensor([1e-2]*self.num_geometrics + [1.0]*self.num_semantics, dtype=torch.float))
        
        self.affine_dxyz = affine_dxyz
        
        self.num_refine = num_refine


    def forward(
        self,
        BGSfeat: torch.Tensor,
        BGS: torch.Tensor,
        BGSembed: torch.Tensor,
        # metas,
    ):
        dfeat = self.layers(BGSfeat + BGSembed) # 1, N, 23
        dfeat = dfeat * self.scale
        
        return torch.cat([BGS[..., :self.num_geometrics]+dfeat[..., :self.num_geometrics], dfeat[..., self.num_geometrics:]], dim=-1)


class FLMOcc(nn.Module):
    def __init__(self, config: Model, rank: int = -1):
        super().__init__()
        self.config = config
        scene.scale_affine = config.scale_affine
        scene.e1e2_affine = config.e1e2_affine
        
        self.voxel_size = config.voxel_size
        
        # 2D encoder
        _config = config.Encoder2D
        ## DAv2
        self.rgb_encoder = RGBEncoder(_config.encoder, _config.path_dav2, rank).load_pretrained_weights()#.lora()
        # delete unused layers after loading pretrained weights
        del self.rgb_encoder.depth_head.scratch.output_conv1
        del self.rgb_encoder.depth_head.scratch.output_conv2
        del self.rgb_encoder.depth_head.scratch.refinenet4.resConfUnit1
        # self.rgb_encoder.pretrained.requires_grad_(False)
        ## MoGeV2
        # self.rgb_encoder = get_moge_encoder(_config.path_mogev2, _config.dim_out_mogev2)
        dim_feat = self.rgb_encoder.features
        
        dim_factor = config.dim_factor
        dim_feat = int(dim_feat * dim_factor)

        num_gs = config.num_gs
        self.register_buffer('gs_id', torch.arange(num_gs, dtype=torch.int32), True)  # [3200]
        
        self.scale_multiplier = config.scale_multiplier

        num_geos = config.num_geometrics
        num_sems = config.num_semantics - int(not config.include_empty)
        self.num_sems = num_sems
        self.scene_model = scene.Superquadrics(
            num_gs=num_gs,
            num_geometrics=num_geos,
            num_semantics=num_sems,
            learnable_init=config.learnable_init,
        )
        
        self.gs_lifter = GaussianLifter(
            num_geometrics=num_geos,
            num_semantics=num_sems,
            dim_feat=dim_feat,
        )
        
        self.gs_former = SparseGaussianFormer(
            gs_encoder=SparseGaussian3DEncoder(
                embed_dims=dim_feat,
                num_geometrics=num_geos,
                num_semantics=num_sems,
            ),
            norm_layer=partial(nn.LayerNorm, dim_feat),
            ffn=partial(FFN, 
                embed_dims=dim_feat, 
                feedforward_channels=dim_feat*2, 
                num_fcs=2,
            ),
            deformable_model=partial(DeformableFeatureAggregation,
                embed_dims=dim_feat,
                num_groups=4,
                num_levels=4,
                num_cams=1,
                attn_drop=0.15,
                kpts_generator=partial(SparseGaussian3DKeyPointsGenerator,
                    embed_dims=dim_feat,
                    num_learnable_pts=0,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                    pc_range=[0,0,0,1,1,1],
                    scale_range=[0.005, 1.0],
                ),
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode="add",
                dim_factor=dim_factor,
                out_of_frustum=config.out_of_frustum,
            ),
            refine_layer=partial(SparseGaussian3DDeltaRefinementModule,
                embed_dims=dim_feat,
                num_geometrics=num_geos,
                num_semantics=num_sems,
                affine_dxyz=config.affine_dxyz,
                num_refine=config.num_refine_blocks,
            ),
            num_block=config.num_refine_blocks,
            attn=partial(MHA3D,
                in_channels=dim_feat,
                num_heads=4,
                custom_freqs=torch.linspace(1., 2**5*torch.pi, int(21/2*dim_factor) // 2),
                pe_dim=int(21/2*dim_factor),
            ),
            operation_order=config.operation_order,
        )


    def init_weights(self):
        for m in self.children():
            if hasattr(m, "init_weight"):
                printd(f"init_weight for {m.__class__.__name__}")
                m.init_weight()
            elif hasattr(m, "init_weights"):
                printd(f"init_weights for {m.__class__.__name__}")
                m.init_weights()


    def forward(self, batch, levels_out=[-1], do_local_agg=False):
        (
            rgbs, BKs_rgb, BWHs_rgb, depths, 
            list_XYZ_query,
            list_voxsize_agg,
            list_xyz_nempty,
            poses,
        ) = (
            batch['rgbs'], batch['Ks'], batch['WHs'], batch['depths'],
            batch['XYZ_query'],
            batch['voxsize_agg'],
            batch['xyz_nempty'],
            batch['poses'],
        )
        B = rgbs.shape[0]

        depths, dict_paths, cls_tokens = self.rgb_encoder(rgbs)
        
        list_featmaps = [dict_paths[i][:, None] for i in range(1,5)]
        BGC, BGF = self.gs_lifter(self.scene_model.GS, list_featmaps)

        LBGC = self.gs_former(BGC, BGF, list_featmaps, BKs_rgb, BWHs_rgb)
        LBGC = torch.stack([BGC] + LBGC)

        tuple_LBGC = self.scene_model.space_transform(LBGC, BKs_rgb[None], BWHs_rgb[None])
        tuple_LBGC_f =  [ LBGC.float() for LBGC in tuple_LBGC ]

        scale_multiplier = self.scale_multiplier
        
        B_LPdensity_sparse = []
        B_LPlogit_sparse  = []
        
        B_Pdensity_dense = []
        B_Plogit_dense  = []
        B_map_out__in = []
        B_mask_input  = []
        for i in range(B):
            Lxyz_sparse = list_xyz_nempty[i][None]  # [1, ?P1, 3]
            voxsize_agg = list_voxsize_agg[i]
            
            tuple_LGC_f = [ LBGC[levels_out, i] if LBGC is not None else None for LBGC in tuple_LBGC_f ]
            Lm, Ls, LR, Lo, Le1e2, Lsem = tuple_LGC_f
            Lo = Lo if self.config.sem_alpha_blend else None
            Lsem = Lsem if self.num_sems > 0 else None
            
            # debug, save init
            # if True:
            if False:
                import ipdb; ipdb.set_trace()
                from utils.misc import save_gs_ply_binary
                
                means = Lm[0].detach().cpu().numpy()
                scales = Ls[0].detach().cpu().numpy()
                from utils.math import rotmat_to_quaternion
                wxyz = rotmat_to_quaternion(LR[0]).detach().cpu().numpy()
                e1e2 = Le1e2[0].detach().cpu().numpy()
                save_gs_ply_binary('debug/debug_sq_init.ply', means, scales, wxyz, e1e2)
                self.debug_saved_init = False
                

            # Global aggregation
            with torch.autocast(device_type='cuda', enabled=False):
                ## Superquadric kernel
                if self.config.use_SQMM:
                    LPdensity_sparse, LPlogits_sparse = SQMM.global_agg_vnsqmm(
                        Lxyz_sparse.float(),
                        Lm, Ls, LR, Le1e2, Lsem,
                    )
                else:
                ## Gaussian kernel
                    LPdensity_sparse, LPlogits_sparse = GMM.global_agg_vngmm(
                        Lxyz_sparse.float(),
                        Lm, Ls, LR, Lo, Lsem,
                    )
            B_LPdensity_sparse.append(LPdensity_sparse)
            B_LPlogit_sparse. append(LPlogits_sparse)
            
            # Local aggregation
            if do_local_agg:
                xyz_dense = list_XYZ_query[i] # [?P2, 3]
                with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                    o = Lo[-1] if Lo is not None else None
                    sem = Lsem[-1] if Lsem is not None else None
                    ## Superquadric kernel
                    if self.config.use_SQMM:
                        Pdensity_dense, Plogit_dense, map_out__in, mask_input = SQMM.local_agg_vnsqmm(
                            xyz_dense.float(),
                            Lm[-1], Ls[-1], LR[-1], Le1e2[-1], sem,
                            float(voxsize_agg), int(scale_multiplier),
                        )
                    else:
                    ## Gaussian kernel
                        Pdensity_dense, Plogit_dense, map_out__in, mask_input = GMM.local_agg_vngmm_point(
                        # Pdensity_dense, Plogit_dense, map_out__in, mask_input = GMM.local_agg_vngmm_point_loop(
                            xyz_dense.float(),
                            Lm[-1], Ls[-1], LR[-1], o, sem,
                            float(voxsize_agg), int(scale_multiplier),
                            # num_loops=64,
                        )
            else:
                Pdensity_dense, Plogit_dense, map_out__in, mask_input = None, None, None, None
            B_Pdensity_dense.append(Pdensity_dense)
            B_Plogit_dense. append(Plogit_dense)
            B_map_out__in.append(map_out__in)
            B_mask_input. append(mask_input)
        
        return {
            'tuple_LBGC':       tuple_LBGC_f,
            'densities_sparse': B_LPdensity_sparse,
            'logits_sparse':    B_LPlogit_sparse,
            'densities_dense':  B_Pdensity_dense,
            'logits_dense':     B_Plogit_dense,
            'map_out__in':      B_map_out__in,
            'list_mask_input':  B_mask_input,
        }
        


if __name__ == "__main__":
    from utils.misc import gpu_timing_repeat
    
    from datasets_.scannet import ScannetDataset as UsedDataset
    from configs import scannet as Config
    from utils.training import to_device
    device_id = 0
    
    dataset = UsedDataset(Config.Dataset, 'val')
    
    # list_idx = list(range(96))
    # list_idx = [0, 1, 2]
    # list_idx = [0, 1]
    list_idx = [0]
    # for i in list_idx:
    #     print(dataset.list_path_rgb[i])
    
    data = [ dataset[i] for i in list_idx ] # 6GB
    # data = [dataset[0]]
    # data = [dataset[i] for i in range(20)] # 15GB
        
    batch = dataset.collate_fn(data)
    batch = to_device(batch, f'cuda:{device_id}', half='bf16')
    
    if Config.Train.mixed_precision:
        model = FLMOcc(Config.Model).cuda(device_id).bfloat16().eval()
    else:
        model = FLMOcc(Config.Model).cuda(device_id).eval()
    
    with torch.no_grad():
        outputs = model(batch, do_local_agg=True ); del outputs

        outputs = model(batch, do_local_agg=True ); del outputs
        outputs = model(batch, do_local_agg=False); del outputs
        time_, outputs = gpu_timing_repeat(model, batch, do_local_agg=False, repeat=100)
        print(f'FLM-Occ Forward: {time_:.2f} s')

    # import pdb; pdb.set_trace()
    