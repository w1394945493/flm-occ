import math
import torch
from .GMM import build_index_maps_point, torch_aggregate1D, E



@torch.compile
def kernel_volume_sq(s, e1e2):
    """
    Compute the integral of the superquadric distribution:

    I = s_x s_y s_z * eps1^2 * eps2 * 2^(3*eps1/2)
        * [Gamma(eps2/2)^2 / Gamma(eps2)]
        * Gamma(eps1) * Gamma(eps1/2)

    Args:
        s: Tensor of shape [..., 3], where last dim = [s_x, s_y, s_z]
        e1e2: Tensor of shape [..., 2], where last dim = [eps1, eps2]

    Returns:
        I: Tensor of shape [...], the integral value for each batch element.
    """
    # Product of scale parameters: s_x * s_y * s_z
    term_scale = s.prod(dim=-1)

    eps1 = e1e2[..., 0]
    eps2 = e1e2[..., 1]

    term_eps = eps1 ** 2 * eps2
    term_pow2 = 2.0 ** (1.5 * eps1)

    lgamma = torch.lgamma
    term_gamma_ratio = torch.exp(2 * lgamma(eps2 / 2.0) - lgamma(eps2))
    term_gamma_prod = torch.exp(lgamma(eps1) + lgamma(eps1 / 2.0))

    I = term_scale * term_eps * term_pow2 * term_gamma_ratio * term_gamma_prod
    return I
    
    
@torch.compile
def kernel_volume_sq_log(s, e1e2):
    """
    Compute the log of the superquadric integral in a numerically stable way.

    Args:
        s: Tensor of shape [..., 3], scale parameters (must be > 0)
        e1e2: Tensor of shape [..., 2], shape parameters (eps1, eps2) (must be > 0)

    Returns:
        log_I: Tensor of shape [...], log of the integral value.
    """
    # Log of product(s) = sum(log(s))
    log_scale = torch.log(s).sum(dim=-1)  # shape [...]

    e1 = e1e2[..., 0]
    e2 = e1e2[..., 1]

    log_eps = 2 * torch.log(e1) + torch.log(e2)
    log_pow2 = 1.5 * e1 * math.log(2.0)

    lgamma = torch.lgamma  # log Gamma
    log_gamma_ratio = 2 * lgamma(e2 / 2.0) - lgamma(e2)
    log_gamma_prod = lgamma(e1) + lgamma(e1 / 2.0)

    log_I = log_scale + log_eps + log_pow2 + log_gamma_ratio + log_gamma_prod
    
    return log_I


def safe_pow(base, exponent):
    """
    # float32
    exp [-inf, 88]
    log [1e-45, 1e38] -> [-103.2789, 87.4982]
    """
    input = torch.log(base.clamp(E)) * exponent
    result = torch.exp(input.clamp(max=84))
    return result


@torch.compile
def global_agg_vnsqmm(x, m, s, R, e1e2, sem):
    """
    x: [..., P, 3]
    m: [..., G, 3]
    s: [..., G, 3]
    R: [..., G, 3, 3], local to camera frame
    e1e2: [..., G, 2]
    """
    d = (x.unsqueeze(-2) - m.unsqueeze(-3))[..., None]        # [B, P, G, 3, 1]
    R = R.unsqueeze(-4).transpose(-2, -1)                     # [B, 1, G, 3, 3], (kernel to camera)^T
    x_sq = R @ d                                              # [B, P, G, 3, 1], turn d to kernel frame
    x_sq = x_sq.squeeze(-1)                                   # [B, P, G, 3]
    s = s.unsqueeze(-3)                                       # [B, 1, G, 3]
    e1e2 = e1e2.unsqueeze(-3)                                 # [B, 1, G, 2]
    
    tmp = (x_sq / s)**2                                       # [B, P, G, 3] 
    # tmp_xy = tmp[..., :2].clamp(E)**(1 / e1e2[..., 1:2])    # [B, P, G, 2]
    # tmp_xy = tmp_xy.sum(-1)                                 # [B, P, G]
    # tmp_z = tmp[..., 2].clamp(E)**(1 / e1e2[..., 0])               # [B, P, G]
    # tmp_sum = tmp_xy.clamp(E)**(e1e2[..., 1]/e1e2[..., 0]) + tmp_z # [B, P, G]
    tmp_xy = safe_pow(tmp[..., :2], 1 / e1e2[..., 1:2])       # [B, P, G, 2]
    tmp_xy = tmp_xy.sum(-1)                                   # [B, P, G]
    tmp_z = safe_pow(tmp[..., 2], 1 / e1e2[..., 0])
    tmp_sum = safe_pow(tmp_xy, e1e2[..., 1]/e1e2[..., 0]) + tmp_z
    density = torch.exp(-0.5 * tmp_sum)                       # [B, P, G]
    density_sum = density.sum(-1)                             # [B, P]
    
    if sem is not None:
        sem = sem.unsqueeze(-3)                               # [B, 1, G, C]
        sem_w = density[..., None] * sem                      # [B, P, G, C]
        sem_wsum = sem_w.sum(-2) / (density_sum[..., None]+E) # [B, P, C]
        
        return density_sum, sem_wsum
    else:
        return density_sum, None


@torch.compile(dynamic=True)
def _local_agg_vnsqmm(x, m, s, R, e1e2):
    """
    x: [M1, 3]
    m: [M1, 3]
    s: [M1, 3]
    R: [M1, 3, 3], local to camera frame
    e1e2: [M1, 2]
    """
    d = (x - m)[..., None]        # [M1, 3, 1]
    R = R.transpose(-2, -1)       # [M1, 3, 3]
    x_sq = R @ d                  # [M1, 3, 1]
    x_sq = x_sq.squeeze(-1)       # [M1, 3]
    
    tmp = (x_sq / s)**2                                   # [M1, 3]
    tmp_xy = tmp[..., :2]**(1 / e1e2[..., 1:2])           # [M1, 2]
    tmp_xy = tmp_xy.sum(-1)                               # [M1]
    tmp_z = tmp[..., 2]**(1 / e1e2[..., 0])               # [M1]
    tmp_sum = tmp_xy**(e1e2[..., 1]/e1e2[..., 0]) + tmp_z # [M1]
    density = torch.exp(-0.5 * tmp_sum)                   # [M1]
    
    return density


@torch.no_grad()
def local_agg_vnsqmm(x, m, s, R, e1e2, sem, voxel_size, scale_factor):
    (
        x_sorted,
        map_query_start__gvox_f,
        map_gs__gvox_f_2,
        index_agg,
        map_out__in,
        mask_input,
        numel_out,
    ) = build_index_maps_point(x, m, s, voxel_size, scale_factor)
    
     # Geometry
    x_sorted = x_sorted[map_query_start__gvox_f]   # [M1, 3]
    m = m [map_gs__gvox_f_2]          # [M1, 3]
    s = s [map_gs__gvox_f_2]          # [M1, 3]
    R = R [map_gs__gvox_f_2]          # [M1, 3, 3]
    e1e2 = e1e2 [map_gs__gvox_f_2]    # [M1, 2]
        
    density = _local_agg_vnsqmm(x_sorted, m, s, R, e1e2) # [M1,]
    density_sum = torch_aggregate1D(numel_out, density, index_agg, reduce='sum') # [M1,]
    
    sem_wsum = None
    if sem is not None:
        sem = sem [map_gs__gvox_f_2] # [M1, C]
        sem_w = sem * density[..., None]
        sem_wsum = torch_aggregate1D(numel_out, sem_w, index_agg, reduce='sum')
        sem_wsum = sem_wsum / (density_sum[..., None] + E)

    return density_sum, sem_wsum, map_out__in, mask_input
