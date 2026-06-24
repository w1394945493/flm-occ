
import logging
printi, printd = logging.getLogger('flm-occ').info, logging.getLogger('flm-occ').debug
import math
from typing import Union, Optional

import torch
from torch import Tensor



E = 1e-8 # epsilon for backward gradients


@torch.compile(dynamic=True)
def encode_key(voxels: Tensor, offset=None):
    if offset is not None:
        voxels = voxels + offset

    # voxels.dtype == int32
    # return (voxels[:, 2]<<20) | (voxels[:, 1]<<10) | (voxels[:, 0])
    return (
        (voxels[:, 2].to(torch.int32) << 20) |
        (voxels[:, 1].to(torch.int32) << 10) |
         voxels[:, 0].to(torch.int32)
    )


@torch.compile(dynamic=True)
def decode_key(key: Tensor, offset: Tensor=None):    
    coords = torch.empty([key.shape[0], 3], dtype=torch.int8, device=key.device)
    coords[:,0] = key & 1023
    coords[:,1] = (key >> 10) & 1023
    coords[:,2] = (key >> 20) & 1023
    if offset is not None:
        coords = coords + offset
    return coords


def build_map_gs__gvox(Scales: Tensor, gs_id: Tensor, voxel_size: float, scale_multiplier: float):
    """_summary_

    Args:
        Scales (tensor): Gaussian Splatting Scales, shape [N, 3].
        gs_id  (tensor): Gaussian index in each view
        voxel_size (float):
        scale_multiplier (int): Define the size of the gaussian voxel grid.

    Returns:
        map_gs__gvox: by gs[map_gs__gvox], we can map each gaussian voxel back to its corresponding gs.
    """
    s_max = Scales.max(dim=-1).values                                         # [N,]
    gs_grid_size = (s_max / voxel_size * scale_multiplier).ceil().int()*2 + 1 # [N,]
    num_gs_to_vox = gs_grid_size**3                                           # [N,]
    # map gs voxels to any gs atributes
    map_gs__gvox = torch.repeat_interleave(gs_id, num_gs_to_vox) # [num_touched,]
    
    # Sort voxels that belong to the same gs, for calculating voxel offsets
    num_gvox = map_gs__gvox.shape[0]
    assert num_gvox < 2**31, "prefix_sum exceeds int32 range"
    
    # calculate_vox_idx_at_each_gs
    offset_1D = torch.empty_like(num_gs_to_vox)
    offset_1D[0] = 0
    offset_1D[1:] = torch.cumsum(num_gs_to_vox[:-1], dim=0)
    idx_vox_at_gs = torch.arange(num_gvox, dtype=torch.int32, device=gs_id.device)
    idx_vox_at_gs -= offset_1D[map_gs__gvox]
    
    return map_gs__gvox, gs_grid_size, idx_vox_at_gs


def voxelize_3dgs(means_gs: Tensor, gs_grid_size: Tensor, voxel_size: float, map_gs__gvox: Tensor,
                  idx_vox_at_gs: Tensor):
    """
    Notes:
        Becareful about the range of means_gs, it may exceed int8 range.
        If we need to check the range, we may need to use int16/int32.
    """
    gvox_grid_size = gs_grid_size[map_gs__gvox] # grid size in which the touched voxels are located
    gvox_grid_size_sq = (gs_grid_size**2)[map_gs__gvox]  # [num_touched]

    # offset = torch.empty(map_gs__gvox.shape[0], 3, dtype=torch.int32, device=means_gs.device)
    offset = torch.empty(map_gs__gvox.shape[0], 3, dtype=torch.int8, device=means_gs.device)
    offset[:, 0] = idx_vox_at_gs // gvox_grid_size_sq
    offset[:, 1] = (idx_vox_at_gs - offset[:, 0] * gvox_grid_size_sq) // gvox_grid_size
    offset[:, 2] = idx_vox_at_gs - offset[:, 0] * gvox_grid_size_sq - offset[:, 1] * gvox_grid_size
    offset -= gvox_grid_size[:, None] // 2  # offset to center the voxel in the grid
    
    # TODO becareful about the range of means_gvox, it may exceed int8 range.
    # We only have [-128, 127] for int8. We may need to use int32.
    means_gvox = (means_gs / voxel_size).floor().to(torch.int8) #.to(torch.int8)  # [N, 3]
    
    offset += means_gvox[map_gs__gvox] # avoid allocating new memory
    gvox = offset

    return gvox


def build_index_maps_voxel(
    XYZ_query: Tensor,
    means: Tensor,
    scale: Tensor,
    voxel_size: float=0.32,
    scale_multiplier: Union[float, int]=3
):
    """Build indexing maps for aggregating information from 3DGS to voxels.
    
    Notes:
        map_xx__yy: mapping from yy to xx
        gvox: gaussian voxels, voxels by voxelizing gaussians
        fvox: voxel in frustum, built from XYZ_range, used as query
        su: sorted and unqiue
        f: in frustum
        gvox_suf: sorted unique gaussian voxels in frustum

    Args:
        XYZ_query (torch.Tensor): Query points in 3D space, shape [Q, 3].
        means (torch.Tensor): Gaussian means, shape [N, 3].
        scale (torch.Tensor): Gaussian scale, shape [N, 3].
        voxel_size (float, optional): Voxel size for voxelization. Defaults to 0.32.
        scale_multiplier (int, optional): Scale multiplier for voxel grid size. Defaults to 3.

    Returns:
        bin_logits (torch.Tensor): Binary logits for each query point, shape [Q-?,].
        feature_wsum (torch.Tensor or None): Weighted sum of features, shape [Q-?, F] if feature is provided, else None.
        map_out__in (torch.Tensor): Map input to output. |output| < |input| because some points are not covered by gs.
        mask_input (torch.Tensor): Mask indicating which input query points were used in aggregation, [Q,]
    """
    device = XYZ_query.device

    # with torch.no_grad(): # Calculating index does not require gradients
    # Voxelize the query points
    # TODO becareful about the range of fvox, it may exceed int8 range.
    # fvox = (XYZ_query / voxel_size).floor().int()
    fvox = (XYZ_query / voxel_size).floor().to(torch.int8)
    min_fvox = fvox.amin(0)

    # Build mapping to index gs voxels back to gs
    gs_id = torch.arange(means.shape[0], dtype=torch.int32, device=device)
        
    map_gs__gvox, gs_grid_size, idx_vox_at_gs = build_map_gs__gvox(scale, gs_id, voxel_size, scale_multiplier)
    # num_gvox = map_gs__gvox.shape[0]  # number of voxels voxelized from all gs
    # assert num_gvox < 2**31, "prefix_sum exceeds int32 range"

    # # Sort voxels that belong to the same gs, for calculating voxel offsets
    # idx_vox_at_gs = calculate_vox_idx_at_each_gs(num_gs_to_vox, num_gvox)

    # Voxelize 3DGS, many voxels are duplicated
    gvox = voxelize_3dgs( # Be careful, gvox is int8 now, [-128, 127]
        means,        # [N, 3]
        gs_grid_size, # [N,]
        voxel_size, 
        map_gs__gvox, #[M,]
        idx_vox_at_gs      #[M,]
    )

    # Find unique gs voxels
    min_gvox = gvox.amin(0)
    offset_vox = torch.stack([min_gvox, min_fvox]).amin(0)
    key_gvox_su, map_gvox_su__gvox= torch.unique(encode_key(gvox, -offset_vox), sorted=True, return_inverse=True)
    del gvox, idx_vox_at_gs

    # Build mapping from unique gsvoxels to frustum voxels
    key_fvox_s, sorter_fvox_s = torch.sort(encode_key(fvox, -offset_vox))
    del fvox
    XYZ_query = XYZ_query[sorter_fvox_s]
    # assert k_frustum_sort.shape[0] == fvox.shape[0], "Frustum voxels should be unique"
    mask_gvox_suf = torch.isin(key_gvox_su, key_fvox_s, assume_unique=True)
    map_fvox_s__gvox_suf = torch.searchsorted(key_fvox_s, key_gvox_su[mask_gvox_suf], out_int32=True)

    # Build mapping from gsvoxels to gs
    mask_gvox_in_frustum = mask_gvox_suf[map_gvox_su__gvox] # M1 = mask_gvox_suf.sum().item()
    map_gs__gvox_f = map_gs__gvox[mask_gvox_in_frustum]     # [M1,]
    del map_gs__gvox

    # Find frustum points that match gs
    map_fvox_s__gvox_su_2 = torch.empty_like(key_gvox_su)
    map_fvox_s__gvox_su_2[mask_gvox_suf] = map_fvox_s__gvox_suf
    map_fvox_s__gvox_f = map_fvox_s__gvox_su_2[map_gvox_su__gvox[mask_gvox_in_frustum]]
    del mask_gvox_in_frustum, map_fvox_s__gvox_su_2, map_gvox_su__gvox, key_gvox_su, mask_gvox_suf

    # Index for aggreagating information
    index_agg = torch.empty(key_fvox_s.shape[0], dtype=torch.int64, device=device) # index必须为long类型
    num_valid_query = map_fvox_s__gvox_suf.shape[0]
    index_agg[map_fvox_s__gvox_suf] = torch.arange(num_valid_query, dtype=torch.int64, device=device)
    index_agg = index_agg[map_fvox_s__gvox_f]
    
    # Build indexes to map input to output
    mask_fvox_s = torch.isin(key_fvox_s, key_fvox_s[map_fvox_s__gvox_suf.unique()], assume_unique=True)
    # map_out__in = sorter_fvox_s[mask_fvox_s]  #
    # mask_input = mask_fvox_s[sorter_fvox_s.sort()[1]] # mask for inputs that are involved in the aggregation
    del map_fvox_s__gvox_suf, key_fvox_s#, sorter_fvox_s, mask_fvox_s
    
    return XYZ_query, map_gs__gvox_f, map_fvox_s__gvox_f, index_agg, sorter_fvox_s, mask_fvox_s, num_valid_query


def torch_aggregate1D(numel: int, src: Tensor, index: Tensor, reduce):
    assert index.ndim == 1, "index must be 1D tensor"
    if src.ndim == 2:
        pholder = src.new_zeros(numel, src.shape[-1])
        index = index[:, None].expand(-1, src.shape[-1])
    elif src.ndim == 1:
        pholder = src.new_zeros(numel)
    elif src.ndim == 3 and src.shape[-1] == 3 and src.shape[-2] == 3: # rotation matrix
        pholder = src.new_zeros(numel, 3, 3)
        index = index[:, None, None].expand(-1, 3, 3)
        
    return pholder.scatter_reduce_(0, index, src, reduce=reduce, include_self=False)
    

@torch.compile
def Information_from_SR(S, R):
    _SR = 1/S.unsqueeze(-1) * R
    I = _SR.transpose(-1, -2) @ _SR # syrk
    # I = torch.einsum('...ij,...ik->...jk', _SR, _SR) # GEMM
    
    return I

norm_const = (2 * math.pi) ** 1.5
def denom_from_S(S):
    return norm_const * S.prod(-1)


@torch.compile(dynamic=True)
def _global_agg_gmm_fwd(x, m, I, o, sem, w=None, sum_feat=True):
    d = (x.unsqueeze(-2) - m.unsqueeze(-3))[..., None]           # [B, P, G, 3, 1]
    # [B, P, G, 1, 3]  @  [B, 1, G, 3, 3]  @  [B, P, G, 3, 1]  =>  [B, P, G, 1, 1]
    ls_term = -0.5 * (d.transpose(-1, -2) @ I.unsqueeze(-4) @ d) # [B, P, G, 1, 1]
    score = torch.exp(ls_term)[..., 0]                           # [B, P, G, 1]
    # HACK for simulating local aggregation
    # sigma = lambda s: math.exp(-0.5 * s**2)
    # mask_zero = score < sigma(3) # approx exp(-0.5*3^2)
    # score = score.masked_fill(mask_zero, 0.0)
    if w is not None:
        score = score * w.unsqueeze(-3)
    del d, ls_term
    
    if sum_feat:
        ret_score = score.sum([-2, -1])                          # [B, P]
    else:
        ret_score = score                                        # [B, P, G, 1]
    
    if sem is not None:
        oscore = score
        if o is not None:
            oscore = o.unsqueeze(-3) * score                     # [B, P, G, 1]
        sum_oscore = oscore.sum(-2, keepdim=True)                # [B, P, 1, 1]
        if sum_feat:
            sum_oscore = sum_oscore[..., 0]                      # [B, P, 1]
            # [B, P, G, 1]  *  [B, 1, G, C]  =>  [B, P, G, C]  =>  [B, P, C]
            sem_w = (oscore * sem.unsqueeze(-3)).sum(-2) / (sum_oscore + E) # [B, P, C]
            del oscore
        
            return ret_score, sem_w
        else:
            # [B, P, G, 1]  *  [B, 1, G, C]   =>   [B, P, G, C]   =>  [B, P, G, C]
            sem_w = (oscore * sem.unsqueeze(-3)) / (sum_oscore + E) # [B, P, G, C]
            del oscore
            
            return ret_score, sem_w
    else:
        return ret_score, None
    

# @torch.compile(dynamic=True) # return None is invalid for torch.compile
def global_agg_vngmm(x, m, s, R, o, sem):
    """
    Kernel volume-weighted Gaussian Mixture Model (GMM) aggregation.
    
    Notes:
        The output likelihoods would not be normalized.
        They are normalized only in the loss function.
    """
    # xyz [B, P, 3]
    # m   [B, G, 3]
    # s   [B, G, 3]
    # R   [B, G, 3, 3] # world to local rotation
    # feature [B, G, C]
    
    # autograd
    I = Information_from_SR(s, R.transpose(-1, -2))
    
    return _global_agg_gmm_fwd(x, m, I, o, sem)


def global_agg_gmm(x, m, s, R, w, sem):
    # xyz [B, P, 3]
    # m   [B, G, 3]
    # s   [B, G, 3]
    # R   [B, G, 3, 3] # world to local rotation
    # w   [B, G, 1]   # weight of each gaussian
    # sem [B, G, C]
    
    # autograd
    I = Information_from_SR(s, R)
    w = w / (w.sum(-2, keepdim=True) + E)
    w = w / (denom_from_S(s).unsqueeze(-1) + E)
    
    return _global_agg_gmm_fwd(x, m, I, None, sem, w)
    

class GlobalGMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, m, I, o, sem):
        """Gaussian function for a single point.
        
        Args:
            x (Tensor): Point to evaluate       [..., P, 3]
            m (Tensor): Mean of the Gaussian    [..., G, 3]
            I (Tensor): Information matrix      [..., G, 3, 3]
            o (Tensor): Opacity of the Gaussian [..., G, 1]
            sem (Tensor): Gaussian logits       [..., G, C]
        """
        ctx.save_for_backward(x, m, I, o, sem)
        
        return _global_agg_gmm_fwd(x, m, I, o, sem)
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    # def backward(ctx, grad_likelihood, grad_logits):
    def backward(ctx, *grad_likelihood_grad_logits):
        """Backward pass for the Gaussian function.
        
        Args:
            grad_likelihood (Tensor): [..., P].
            grad_logits (Tensor):     [..., P, C].
        """
        sem = ctx.saved_tensors[-1]
        if sem is not None:
            return _global_agg_gmm_bwd(ctx.saved_tensors, *grad_likelihood_grad_logits)
        else:
            return _global_agg_gmm_bwd(ctx.saved_tensors, grad_likelihood_grad_logits[0], None)


# @torch.compile(dynamic=True)
def _global_agg_gmm_bwd(saved_tensors, grad_likelihood, grad_logits):
    x, m, I, o, sem = saved_tensors
    # dummy grad for debug
    # grad_m = torch.zeros_like(m)
    # grad_I = torch.zeros_like(I)
    # grad_o = torch.zeros_like(o) if o is not None else None
    # grad_sem = torch.zeros_like(sem) if sem is not None else None

    # Forward pass
    score, sem_w = _global_agg_gmm_fwd(x, m, I, o, sem, sum_feat=False)
                                                 # [B, P, G, 1], [B, P, G, C]        
    # Geometry Gradients
    grad_likelihood = grad_likelihood[..., None, None]         # [B, P, 1, 1]
    grad_exp = grad_likelihood
    if sem is not None:
        oscore = score
        if o is not None:
            o = o.unsqueeze(-3)                                # [B, 1, G, 1]
            oscore = o * score                                 # [B, P, G, 1]
            sum_os = oscore.sum(-2, keepdim=True)              # [B, P, 1, 1]
            o_d_sum_os = o / (sum_os + E)                      # [B, P, G, 1]
        else:
            sum_os = oscore.sum(-2, keepdim=True)              # [B, P, 1, 1]
            o_d_sum_os = 1 / (sum_os + E)                      # [B, P, G, 1]
        sem_wsum = sem_w.sum(-2, keepdim=True)                 # [B, P, 1, C]
        sem_s_sem_ws = sem.unsqueeze(-3) - sem_wsum            # [B, P, G, C]
        grad_logits = grad_logits.unsqueeze(-2)                # [B, P, 1, C]
        grad_sem_w = grad_logits * o_d_sum_os * sem_s_sem_ws   # [B, P, G, C]
        grad_exp = grad_exp + grad_sem_w.sum(-1, keepdim=True) # [B, P, G, 1]
        del sem_w, o_d_sum_os, grad_sem_w, sem_wsum
    
    grad_exp = grad_exp[..., None]                             # [B, P, G, 1, 1]
    I = I.unsqueeze(-4)                                        # [B, 1, G, 3, 3]
    d = (x.unsqueeze(-2) - m.unsqueeze(-3))[..., None]         # [B, P, G, 3, 1]

    grad_m = (
        grad_exp * (score[..., None] * I @ d)
    ).sum([-4, -1])                         # [B, P, G, 3, 1] => [B, G, 3]

    grad_I = (
        grad_exp * -0.5 * score[..., None] * d @ d.transpose(-1, -2)
    ).sum(dim=-4)                           # [B, P, G, 3, 3] => [B, G, 3, 3]
    del d, I, grad_exp
    
    # Semantics Gradients
    grad_o = None
    grad_sem = None
    if sem is not None:
        if o is not None:
            s_d_sum_os = score / (sum_os + E)                  # [B, P, G, 1]
            grad_o = (grad_logits * s_d_sum_os * sem_s_sem_ws) # [B, P, G, C]
            grad_o = grad_o.sum([-3,-1])[..., None]            # [B,    G  1]
            del s_d_sum_os, sem_s_sem_ws
        weight = oscore / (sum_os + E)                         # [B, P, G, 1]
        grad_sem = (grad_logits * weight).sum(-3)              # [B,    G, C]
        del weight, oscore, sum_os
    
    return None, grad_m, grad_I, grad_o, grad_sem, None
    
    
def global_agg_vngmm_mangrad(points, m, s, R, o, sem):
    # manual grad
    # _SR = torch.diag_embed(1/s, dim1=-2, dim2=-1) @ R
    # Info = _SR.transpose(-1, -2) @ _SR
    Info = Information_from_SR(s, R)
    
    return GlobalGMMFunction.apply(points, m, Info, o, sem)


@torch.compile(dynamic=True)
def _local_agg_gmm_point(x, m, I, w=None):
    
    d = x - m  # [M1, 3]

    # [M1, 1, 3] @ [M1, 3, 3] @ [M1, 3, 1] => [M1, 1, 1] => [M1,]
    # ls_term = -0.5 * (d[:,None,:] @ I @ d[..., None])[:, 0, 0]
    # Below is way faster than the above
    result = (
        # 对角线项
        d[:, 0]**2 * I[:, 0, 0] +
        d[:, 1]**2 * I[:, 1, 1] + 
        d[:, 2]**2 * I[:, 2, 2] +
        # 非对角线项*2
        2 * d[:, 0] * I[:, 0, 1] * d[:, 1] +
        2 * d[:, 0] * I[:, 0, 2] * d[:, 2] +
        2 * d[:, 1] * I[:, 1, 2] * d[:, 2]
    )
    ls_term = -0.5 * result
    score = torch.exp(ls_term) # [M1,]
    if w is not None:
        score = score * w[:, 0]
    
    return score, d


class LocalGMMPointAggregator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, map_query_start__gvox_f, m, I, w, o, sem, map_gs__gvox_f_2, index_agg, numel_out):
        """Forward pass for local GMM point aggregation.
        """
        
        save_tensor = [x, map_query_start__gvox_f, m, I, w, o, sem, map_gs__gvox_f_2, index_agg, numel_out]
        
        # Geometry
        x = x [map_query_start__gvox_f] # [M1, 3]
        m = m [map_gs__gvox_f_2]        # [M1, 3]
        I = I [map_gs__gvox_f_2]        # [M1, 3, 3]
        if w is not None:
            w = w [map_gs__gvox_f_2]    # [M1, 1]
        
        score, d = _local_agg_gmm_point(x, m, I, w)
        del x, m, I
        if w is not None:
            del w
        
        likelihood = torch_aggregate1D(numel_out, score, index_agg, reduce='sum')
        
        # Semantics
        sem_wsum = None
        score = score[:, None]          # [M1, 1]
        if sem is not None:
            sem = sem[map_gs__gvox_f_2] # [M1, C]
            oscore = score
            if o is not None:
                o = o[map_gs__gvox_f_2]
                oscore = o * score
                sum_weights = torch_aggregate1D(numel_out, oscore, index_agg, reduce='sum') # [numel_out, 1]
                # [numel_out, C]
                sem_w = oscore * sem
                sem_wsum = torch_aggregate1D(numel_out, sem_w, index_agg, reduce='sum') / (sum_weights + E)
            else:
                # [numel_out, C]
                sem_w = oscore * sem
                sem_wsum = torch_aggregate1D(numel_out, sem_w, index_agg, reduce='sum') \
                    / (likelihood[..., None] + E)
        
        save_tensor += [score, d, sem_wsum]
        ctx.save_for_backward(*save_tensor)
        
        return likelihood, sem_wsum
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_likelihood, grad_logits):
        """Backward pass for local GMM point aggregation.
        Be careful, vanilla gmm are not supported yet. Only volume-normalized gmm is supported.
        """
        
        (
            x, map_query_start__gvox_f, m, I, w, o, sem, map_gs__gvox_f_2, index_agg, numel_out,
            score, d, sem_wsum
        ) = ctx.saved_tensors
        num_gs = m.shape[0]
        
        # # dummy grad for debug
        # grad_m = torch.zeros_like(m)
        # grad_I = torch.zeros_like(I)
        grad_w = torch.zeros_like(w) if w is not None else None
        # grad_o = torch.zeros_like(o) if o is not None else None
        # grad_sem = torch.zeros_like(sem) if sem is not None else None
        # sem_wsum = sem_wsum[index_agg] if sem_wsum is not None else None
        
        # ------------- Geometry -------------
        x = x [map_query_start__gvox_f] # [M1, 3]
        m = m [map_gs__gvox_f_2]        # [M1, 3]
        I = I [map_gs__gvox_f_2]        # [M1, 3, 3]
        # score [M1, 1]
        # score, d = _local_agg_gmm_point(x, m, I)
        
        map_gs__gvox_f_2 = map_gs__gvox_f_2.long()
        grad_exp = grad_likelihood[index_agg, None] # [M1, 1]
        if sem is not None:
            oscore = score  # [M1, 1]
            if o is not None:
                o = o[map_gs__gvox_f_2]
                oscore = o * score
                sum_os = torch_aggregate1D(numel_out, oscore, index_agg, reduce='sum') # [numel_out, 1]
                sum_os = sum_os[index_agg]                                             # [M1, 1]
                o_d_sum_os = o / (sum_os + E)                                          # [M1, 1]
            else:
                sum_os = torch_aggregate1D(numel_out, oscore, index_agg, reduce='sum') # [numel_out, 1]
                sum_os = sum_os[index_agg]                                             # [M1, 1]
                o_d_sum_os = 1 / (sum_os + E)                                          # [M1, 1]
            sem_s_sem_ws = sem[map_gs__gvox_f_2] - sem_wsum                            # [M1, C]
            grad_logits = grad_logits[index_agg]                                       # [M1, C]
            grad_sem_w = grad_logits * o_d_sum_os * sem_s_sem_ws                       # [M1, C]
            grad_exp = grad_exp + grad_sem_w.sum(-1, keepdim=True)                     # [M1, 1]

        grad_m = (grad_exp * score) * (I @ d[..., None])[..., 0]  # [M1, 3]
        grad_m = torch_aggregate1D(num_gs, grad_m, map_gs__gvox_f_2, reduce='sum')
        
        grad_I = (grad_exp * -0.5 * score)[..., None] * (d[..., None] @ d[:, None, :]) # [M1, 3, 3]
        grad_I = torch_aggregate1D(num_gs, grad_I, map_gs__gvox_f_2, reduce='sum')

        # # ------------- Semantics -------------
        if sem is not None:
            if o is not None:
                s_d_sum_os = score / (sum_os + E)                                      # [M1, 1]
                grad_o = grad_logits * sem_wsum * s_d_sum_os
                grad_o = grad_o.sum(-1)                                                # [M1,]
                grad_o = torch_aggregate1D(num_gs, grad_o, map_gs__gvox_f_2, reduce='sum')
            weight = oscore / (sum_os + E)                                             # [M1, 1]

            grad_sem = (grad_logits * weight)
            grad_sem = torch_aggregate1D(num_gs, grad_sem, map_gs__gvox_f_2, reduce='sum')

        return None, None, grad_m, grad_I, grad_w, grad_o, grad_sem, None, None, None


def local_agg_vngmm_voxel(
    x: Tensor,
    m: Tensor,
    s: Tensor,
    R: Tensor,
    w: Tensor,
    sem: Tensor,
    voxel_size: float=0.32,
    scale_muliplier: Union[float, int]=3
):
    """
    Used when voxels are represented in camera frame
    so that the grid is axis-aligned with camera axes.
    """
    (
        x_sorted, # x is sorted
        map_gs__gvox_f,
        map_fvox_s__gvox_f,
        index_agg,
        map_out__in,
        mask_input,
        num_valid_query
    ) = build_index_maps_voxel(
        x, m, s, voxel_size, scale_muliplier
    )
    
    if num_valid_query == 0:
        likelihood = torch.zeros((0,), dtype=x.dtype, device=x.device)
        if sem is not None:
            semantics = torch.zeros((0, sem.shape[-1]), dtype=sem.dtype, device=sem.device)
        else:
            semantics = None
        return likelihood, semantics, map_out__in, mask_input
    
    I = Information_from_SR(s, R)
    
    likelihood, semantics = LocalGMMPointAggregator.apply(
        x_sorted, map_fvox_s__gvox_f,
        m, I, None, None, sem, map_gs__gvox_f,
        index_agg, num_valid_query
    )
    
    return likelihood, semantics, map_out__in, mask_input


# --------------------- point aggregation ---------------------

def build_index_maps_point(
    XYZ_query: Tensor,
    means: Tensor,
    scale: Tensor,
    voxel_size: float,
    scale_multiplier: Union[float, int]=3
):
    """Build indexing maps for aggregating information from 3DGS to query points.
    
    Notes:
        map_xx__yy: mapping from yy to xx
        gvox: gaussian voxels, voxels by voxelizing gaussians
        fvox: voxel in frustum, built from XYZ_range, used as query
        su: sorted and unqiue
        f: in frustum
        gvox_suf: sorted unique gaussian voxels that are in frustum
        
        Calculations with int8 key are not supported by CPU!
        
        For point aggregation, frustum points are actually query points and query points are not unique.

    Args:
        XYZ_query (torch.Tensor): Query points in 3D space, shape [Q, 3].
        means (torch.Tensor): Gaussian means, shape [N, 3].
        scale (torch.Tensor): Gaussian scale, shape [N, 3].
        voxel_size (float): Voxel size for voxelization. Defaults to 0.32.
        scale_multiplier (int, optional): Scale multiplier for voxel grid size. Defaults to 3.

    Returns:
        XYZ_query               (torch.Tensor): Sorted query points, shape [Q, 3].
        map_query_start__gvox_f (torch.Tensor): Map gs voxels to query points, shape [M1,].
        map_gs__gvox_f_2        (torch.Tensor): Map gs voxels to gs, shape [M1,].
        index_agg               (torch.Tensor): Index for aggregating gs, shape [M1,].
        mask_query              (torch.Tensor): Mask for query points involved in aggregation, shape [Q,].
        numel_valid_query       (torch.Tensor): Number of points involved in aggregation.
    """
    device = XYZ_query.device

    # Voxelize the query points
    fvox = (XYZ_query / voxel_size).floor().to(torch.int8) # cpu的torch.compile不支持int8
    min_fvox = fvox.amin(0)
    max_fvox = fvox.amax(0)
    
    # Build mapping to index gs voxels back to gs
    gs_id = torch.arange(means.shape[0], dtype=torch.int32, device=device)

    map_gs__gvox, gs_grid_size, idx_vox_at_gs = build_map_gs__gvox(scale, gs_id, voxel_size, scale_multiplier)
    # num_gvox = map_gs__gvox.shape[0]  # number of voxels voxelized from all gs
    # assert num_gvox < 2**31, "prefix_sum exceeds int32 range"

    # Sort voxels that belong to the same gs, for calculating voxel offsets
    # idx_vox_at_gs = calculate_vox_idx_at_each_gs(num_gs_to_vox, num_gvox)

    # Voxelize 3DGS, many voxels are duplicated
    gvox = voxelize_3dgs( # Be careful, gvox is int8 now, [-128, 127]
        means,        # [N, 3]
        gs_grid_size, # [N,]
        voxel_size, 
        map_gs__gvox, #[M,]
        idx_vox_at_gs #[M,]
    )
    min_gvox = gvox.amin(0)
    max_gvox = gvox.amax(0)

    # Find unique gs voxels
    voxel_min = torch.stack([min_gvox, min_fvox]).amin(0)
    voxel_max = torch.stack([max_gvox, max_fvox]).amax(0)
    voxel_range = voxel_max - voxel_min + 1
    assert (voxel_range.int() <= 256).all(), f"voxel range exceeds int8 range: {voxel_range}"
    key_fvox_s, sorter_fvox_s = encode_key(fvox, -voxel_min).sort()
    key_fvox_su, map_fvox_su__fvox_s, counts_fvox_su = torch.unique_consecutive(
        key_fvox_s, return_inverse=True, return_counts=True)
    XYZ_query = XYZ_query[sorter_fvox_s]
    
    # region time consuming part
    key_gvox_s, sorter_gvox_s = encode_key(gvox, -voxel_min).sort()
    map_gs__gvox = map_gs__gvox[sorter_gvox_s]
    key_gvox_su, map_gvox_su__gvox = torch.unique_consecutive(key_gvox_s, return_inverse=True)
    del key_gvox_s, sorter_gvox_s, gvox, idx_vox_at_gs
    # endregion

    # Build mapping from unique gsvoxels to frustum voxels
    mask_gvox_suf = torch.isin(key_gvox_su, key_fvox_su, assume_unique=True)
    map_fvox_s__gvox_suf = torch.searchsorted(key_fvox_su, key_gvox_su[mask_gvox_suf], out_int32=True)

    # Find mask that indicates which query points are used in aggregation
    mask_fvox_su = torch.isin(key_fvox_su, key_fvox_su[map_fvox_s__gvox_suf.unique()], assume_unique=True)
    mask_fvox_s = mask_fvox_su[map_fvox_su__fvox_s]
    
    # handle situation where no points are covered by Gaussians
    if mask_fvox_s.sum() == 0:
        # print('Get no points covered by Gaussians in local_agg_gmm_point')
        return (
            XYZ_query,
            torch.empty((0,), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
            sorter_fvox_s, mask_fvox_s, 0
        )
        
    # Build mapping from gsvoxels to gs
    mask_gvox_in_frustum = mask_gvox_suf[map_gvox_su__gvox] # M1 = mask_gvox_suf.sum().item()
    map_gs__gvox_f = map_gs__gvox[mask_gvox_in_frustum]     # [M1,]

    # Find frustum points that match gs
    map_fvox_su__gvox_su_2 = torch.empty_like(key_gvox_su)
    map_fvox_su__gvox_su_2[mask_gvox_suf] = map_fvox_s__gvox_suf
    map_fvox_su__gvox_f = map_fvox_su__gvox_su_2[map_gvox_su__gvox[mask_gvox_in_frustum]]
    del mask_gvox_in_frustum, map_gvox_su__gvox, map_fvox_su__gvox_su_2, map_gs__gvox

    # map voxel back to points
    counts_map_fvox_su__gvox_f = counts_fvox_su[map_fvox_su__gvox_f]
    idx_counts_map_fvox_su__gvox_f = torch.repeat_interleave(
        torch.arange(counts_map_fvox_su__gvox_f.shape[0], dtype=torch.int32, device=device),
        counts_map_fvox_su__gvox_f
    )
    
    offset = torch.empty_like(counts_map_fvox_su__gvox_f, dtype=torch.int32, device=device)
        
    offset[0] = 0
    offset[1:] = torch.cumsum(counts_map_fvox_su__gvox_f[:-1], dim=0)  # [M1,]
    sum_counts_map_fvox_su__gvox_f = offset[-1] + counts_map_fvox_su__gvox_f[-1]
    
    offset = offset[idx_counts_map_fvox_su__gvox_f]
    pts_rank_in_fvox = torch.arange(sum_counts_map_fvox_su__gvox_f, dtype=torch.int32, device=device) - offset
    del offset, counts_map_fvox_su__gvox_f

    start_idx_query = torch.empty_like(counts_fvox_su, dtype=torch.int32, device=device)
    start_idx_query[0] = 0
    start_idx_query[1:] = torch.cumsum(counts_fvox_su[:-1], dim=0)

    # region time consuming part
    map_query_start__gvox_f = start_idx_query[map_fvox_su__gvox_f]
    map_query__gvox_f = map_query_start__gvox_f[idx_counts_map_fvox_su__gvox_f] + pts_rank_in_fvox
    del map_query_start__gvox_f, pts_rank_in_fvox, map_fvox_su__gvox_f, start_idx_query
    map_gs__gvox_f_2 = map_gs__gvox_f[idx_counts_map_fvox_su__gvox_f]
    del map_gs__gvox_f, idx_counts_map_fvox_su__gvox_f
    # endregion
    
    # Index for aggreagating information
    index_agg = torch.empty(XYZ_query.shape[0], dtype=torch.int64, device=device) # index必须为long类型
    numel_valid_query = mask_fvox_s.sum()
    index_agg[mask_fvox_s] = torch.arange(numel_valid_query, dtype=torch.int64, device=device)
    index_agg = index_agg[map_query__gvox_f]

    return XYZ_query, map_query__gvox_f, map_gs__gvox_f_2, index_agg, sorter_fvox_s, mask_fvox_s, numel_valid_query


def local_agg_vngmm_point(
    x, m, s, R, o, sem,
    voxel_size,
    scale_multiplier,
):
    """
    Volume-normalized Gaussian mixture model (GMM) aggregation.
    
    Only support batch size = 1 for now.
    """
    (
        x_sorted,
        map_query_start__gvox_f,
        map_gs__gvox_f_2,
        index_agg,
        map_out__in,
        mask_input,
        numel_valid_query
    ) = build_index_maps_point(x, m, s, voxel_size, scale_multiplier)
    
    if numel_valid_query == 0:
        # no points are covered by Gaussians
        likelihood = torch.zeros((0,), device=x.device)
        if sem is not None:
            semantics = torch.zeros((0, sem.shape[1]), device=x.device)
        else:
            semantics = None
        return likelihood, semantics, map_out__in, mask_input

    I = Information_from_SR(s, R.transpose(-1, -2))

    likelihood, semantics = LocalGMMPointAggregator.apply(
        x_sorted, map_query_start__gvox_f,
        m, I, None, o, sem, map_gs__gvox_f_2,
        index_agg, numel_valid_query
    )
    
    return likelihood, semantics, map_out__in, mask_input


def local_agg_vngmm_point_loop(
    x, m, s, R, o, sem,
    voxel_size,
    scale_multiplier,
    num_loops: int=10
):
    num_pts = x.shape[0]
    chunk_size = int(math.ceil(num_pts / num_loops))
    list_likelihood  = []
    list_semantics   = []
    list_map_out__in = []
    list_mask_input  = []
    for i in range(num_loops):
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, num_pts)
        x_chunk = x[start_idx:end_idx]
        likelihood, semantics, map_out__in, mask_input = local_agg_vngmm_point(
            x_chunk, m, s, R, o, sem,
            voxel_size,
            scale_multiplier,
        )
        list_likelihood.append(likelihood)
        list_semantics.append(semantics)
        list_map_out__in.append(map_out__in + start_idx)
        list_mask_input.append(mask_input)
    likelihood  = torch.cat(list_likelihood, dim=0)
    semantics   = torch.cat(list_semantics, dim=0)
    map_out__in = torch.cat(list_map_out__in, dim=0)
    mask_input  = torch.cat(list_mask_input, dim=0)
    
    return likelihood, semantics, map_out__in, mask_input


def local_agg_gmm_point(
    x, m, s, R, w, sem,
    voxel_size,
    scale_multiplier,
):
    """
    Only support batch size = 1 for now.
    """
    (
        x_sorted,
        map_query_start__gvox_f,
        map_gs__gvox_f_2,
        index_agg,
        map_out__in,
        mask_input,
        numel_valid_query
    ) = build_index_maps_point(x, m, s, voxel_size, scale_multiplier)

    I = Information_from_SR(s, R)
    w = w / (w.sum(-2, keepdim=True) + E)
    w = w / (denom_from_S(s).unsqueeze(-1) + E)

    likelihood, semantics = LocalGMMPointAggregator.apply(
        x_sorted, map_query_start__gvox_f,
        m, I, w, None, sem, map_gs__gvox_f_2,
        index_agg, numel_valid_query
    )
    
    return likelihood, semantics, map_out__in, mask_input
