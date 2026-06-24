import os
from inspect import isclass, getsource




# dataset setup
list_dir_datasets = [
    '/', # dummy
]
dir_dataset = [ p for p in list_dir_datasets if os.path.exists(p) ][0]


list_path_dav2 = [
    '/', # dummy
]
path_dav2 = [ p for p in list_path_dav2 if os.path.exists(p) ][0]



class Base:
    @classmethod
    def tostring(cls, indent='|   ', level=0):
        lines = []
        lines.append(cls.__name__+': {')
        for key, value in vars(cls).items():
            if not key.startswith("__"):
                if isclass(value):
                    lines += value.tostring(indent, level + 1)
                elif callable(value):
                    lines.append(indent + f'{key}: {getsource(value).strip()}')
                elif isinstance(value, (list, tuple)):
                    str_ = "\n"+f'{indent}    ' + (",\n"+f'{indent}    ').join(map(str, value))
                    lines.append(indent + f'{key}: [{str_},\n{indent}]')
                else:
                    if isinstance(value, str) and '\n' in value:
                        value = value.replace('\n', '\n' + indent)
                    lines.append(indent + f'{key}: {value}')
        lines.append('}')
        
        if level == 0:
            return '\n'.join(indent * level + line for line in lines)
        else:
            return [ indent + l for l in lines ]


class Encoder2D(Base):
    encoder = 'vitb'
    path_dav2 = path_dav2
    
    
class Model(Base):
    depth_min = 0.29
    depth_max = 10.24
    use_SQMM = False  # whether to use Superquadric Mixture Model
    
    num_gs = 512
    num_geometrics = 3+3+4+1+2  # means(3) + scales(3) + quaternions(4) + opa(1) + uv(2)
    num_semantics = 2
    sem_alpha_blend = True
    include_empty = False
    scale_multiplier = 3  # control gaussian voxel range
    voxel_size = 0.32
    learnable_init = False # whether to learn the initial GS anchors
    scale_init_factor = 10 # used in scene.py
    
    out_of_frustum = False
    scale_affine = lambda x: x * 0.5 + 0.01
    e1e2_affine = lambda x: x * 1.9 + 0.1
    
    Encoder2D = Encoder2D
    dim_factor = 2
    
    affine_dxyz = lambda x: x * 0.1
    
    # refine layers
    num_refine_blocks = 3    
    operation_order = [
        "deformable",
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
    ] * num_refine_blocks


class Dataset(Base):
    dir_dataset = dir_dataset
    depth_min = Model.depth_min
    depth_max = Model.depth_max
    
    voxel_size = Model.voxel_size
    num_semantics = Model.num_semantics # 11 + 1 for empty
    include_empty = Model.include_empty # for labels_nempty
    
    random_offset = True

    debug = False


class Train(Base):
    notes = """
    """

    check_anomaly = False  # whether to set_detect_anomaly

    random_seed = 168
    # devices = '0,1,2,3'
    devices = 1
    mixed_precision = False
    exp_dir = 'exp/250826_1-train_neck_conv2d-nll_prod-extrasq'
    
    # dataloader
    num_workers = 24
    prefetch_factor = 2
    persistent_workers = True
    batch_size = 24
    
    # Loss
    num_levels = Model.num_refine_blocks + 1
    levels_out = [-1]         # which levels to output loss
    weights_gmm_level = [1.0] # weights for gmm loss at different levels
    lambda_gmm = 1.0
    lambda_lreg = 0.2
    lambda_ce = 0.0
    
    total_iters = 10000
    
    log_iters = 100
    ckpt_iters = 1000
    val_iters = 1000
    
    grad_norm = 35
    
    # optimizer
    betas = (0.9, 0.999) # unused as it is exactly the default value
    # training_ckpt = 'exp/250814_3-log_contraction-dxyz4-trainDPT-nll_prod-20epoch_once/ckpt/008099.pth'
    finetune = False
    lr = 5e-4
    pretrained_lr = lr / 10
    weight_decay = 0.01 # 0.01 is the default value
    betas = (0.9, 0.999)
    check_grad_nan_inf = False  # check for NaN/Inf in gradients

    training_ckpt = None
    swanlab_id = None
    use_swanlab = True

    # scheduler
    use_scheduler = False
    stype: str = 'warmup_cosine'
    warmup_epochs: float = 1
    cosine_epochs: float = 10
    f_min = 0.1,
    f_max = 1.0,
    
    

if __name__ == '__main__':
    # print the config
    print('\n' + Train.tostring())
    print('\n' + Dataset.tostring())
    print('\n' + Model.tostring())
