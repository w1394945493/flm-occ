import os

from .base import Model, Dataset, Train



# dataset setup
list_dir_datasets = [
    '/path/to/occscannet',
]
Dataset.dir_dataset = [ p for p in list_dir_datasets if os.path.exists(p) ][0]
Dataset.depth_min = 0.1
Dataset.depth_max = 6.9
Dataset.num_semantics = 12 # 11 + 1 for empty
Dataset.include_empty = False # whether to include empty class in labels_query
Dataset.color_jitter = (0.4, 0.4, 0.4) # adopt from NDC-scene
Dataset.fliplr = 0.5
Dataset.random_offset = True
# Dataset.debug = True
Dataset.data_tg = 'mini'


list_path_dav2 = [
    '/path/to/depth_anything_v2',
]
path_dav2 = [ p for p in list_path_dav2 if os.path.exists(p) ][0]
# path_dav2 += '/checkpoints/depth_anything_v2_vitb.pth'
path_dav2 += '/checkpoints/finetune_scannet_depthanythingv2.pth'
Model.Encoder2D.path_dav2 = path_dav2
Model.dim_factor = 2

# Model.Encoder2D.encoder = 'vitl'
# Model.use_SQMM = True
Model.num_gs = 256
Model.scale_init_factor = 10
Model.num_semantics = Dataset.num_semantics
Model.sem_alpha_blend = False
Model.include_empty = Dataset.include_empty
Model.depth_min = Dataset.depth_min
Model.depth_max = Dataset.depth_max
Model.scale_affine = lambda x: x + 0.02
Model.e1e2_affine = lambda x: x * 1.9 + 0.1
Model.affine_dxyz = lambda x: x * 0.1
Model.learnable_init = True
Model.num_refine_blocks = 4
Model.operation_order = [
        "deformable",
        "norm",
        "identity",
        "ffn",
        "add",
        "norm",
        "identity",
        "attn",
        "add",
        "norm",
        "identity",
        "ffn",
        "add",
        "norm",
        "refine",
] * Model.num_refine_blocks

Train.notes = """
Unconstraint refinement
"""


# Train.check_anomaly = True
Train.random_seed = 666
Train.notes = "Test run"
Train.mixed_precision = True
Train.num_workers = 8
Train.prefetch_factor = 1
Train.batch_size = 32
Train.devices = 4
MM = 'SQ' if Model.use_SQMM else 'G'
Train.exp_dir = f'exp/test'

Train.lambda_gmm = 1.0
Train.lambda_lreg = 0.1
Train.lambda_ce = 1.0


Train.total_iters = 15e3
Train.log_iters = 100
Train.ckpt_iters = 2000
Train.val_iters = 1000


Train.lr = 5e-4
Train.pretrained_lr = Train.lr / 10
Train.betas = (0.85, 0.95)
Train.weight_decay = 0.01
Train.grad_norm = -1

# Train.training_ckpt = 'exp/test/ckpt/000007.pth'
# Train.swanlab_id = 'XXXX'
# Train.use_swanlab = False
Train.swanlab_proj = 'FLM_Occ'

# scheduler
Train.use_scheduler = True
Train.stype: str = 'warmup_cosine'
# Train.stype: str = 'warmup_constant'
Train.warmup_iters: int = 1000
Train.cosine_iters: int = 15e3
Train.f_min = 0.1
Train.f_max = 1.0
