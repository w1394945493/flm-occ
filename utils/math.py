import torch
import torch.nn.functional as F

SIGMOID_MAX = 9.21 # sigmoid(9.21) = 1 - 1e-4
LOGIT_MIN = 1 - 0.9999
# SIGMOID_MAX = 13.8155 # sigmoid(-13.8155) ~= 1e-6
# LOGIT_MIN = 1e-6


# @torch.compile(dynamic=True) # dynamic because xyz/s have different dimensions
def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX)
    return torch.sigmoid(tensor)

# @torch.compile(dynamic=True)
def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, LOGIT_MIN, 1 - LOGIT_MIN)
    return torch.log(tensor / (1 - tensor))

# @torch.compile(dynamic=True)
def safe_tanh(tensor):
    return 2 * safe_sigmoid(tensor) - 1

def safe_inverse_tanh(tensor):
    return safe_inverse_sigmoid((tensor + 1) / 2)

def coordinate_system(ray):
    """以ray为z轴建立右手正交坐标系，refer to "Building an Orthonormal Basis, Revisited" Listing 3.

    /* Based on "Building an Orthonormal Basis, Revisited" by
       Tom Duff, James Burgess, Per Christensen,
       Christophe Hery, Andrew Kensler, Max Liani,
       and Ryusuke Villemin (JCGT Vol 6, No 1, 2017) */
       
    Args:
        ray (ndarray): Nx3, camera ray, light path

    Raises:
        Exception: z coordinate of the ray must larger than 0.

    Returns:
        x (ndarray): Nx3
        y (ndarray): Nx3 
    """
    if (ray[:,2] == 0).any():
        raise Exception("Invalid z coordinate. Require z > 0.")

    sign = torch.sign(ray[:,2])

    a = -1 / (sign + ray[:,2])
    b = ray[:,0] * ray[:,1] * a
    x = torch.column_stack([1 + sign * ray[:,0] ** 2 * a, sign * b, -sign * ray[:,0]])
    y = torch.column_stack([b, sign + ray[:,1] ** 2 * a, -ray[:,1]])
    # x = x / np.linalg.norm(x)
    # y = y / np.linalg.norm(y)

    return x, y

def cartesian(anchor, pc_range):
    xyz = safe_sigmoid(anchor[..., :3])
    xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz


# @torch.compile
def normalize_intrinsics(BK, BWH):
    """
    Notes:
    same as PRoPE
    
    Args:
        BK (tensor):  [B, ..., 3, 3] intrinsic matrix
        BHW (tensor): [B, ..., 2] image height and width

    Returns:
        BK: normalized intrinsic matrix
    """
    # fx:=fx/W
    # fy:=fy/H
    # cx:=cx/W-0.5
    # cy:=cy/H-0.5
    # image plane [-0.5,0.5]^2
    BK = BK.clone()
    W = BWH[..., 0]
    H = BWH[..., 1]
    BK[..., 0, 0] /= W
    BK[..., 1, 1] /= H
    BK[..., 0, 2] = BK[..., 0, 2] / W - 0.5
    BK[..., 1, 2] = BK[..., 1, 2] / H - 0.5
    
    return BK


# @torch.compile
def rotation_from_quaternion(quaternions):
    """
    Args:
        quaternions: 形状为 [..., 4] 的单位四元数张量 (w, x, y, z)
        
    Returns:
        torch.Tensor: 形状为 [..., 3, 3] 的旋转矩阵
    """
    quaternions = F.normalize(quaternions, dim=-1)
    w, x, y, z = quaternions.unbind(-1)

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
 
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)

    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    rotation_matrices = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)
    
    return rotation_matrices


def batch_quaternion_multiply(q_target, q_src):
    """
    R_target @ R_src
    Args:
        q_target (torch.Tensor): [..., 4] 四元数 (w, x, y, z)
        q_src (torch.Tensor): [..., 4] 四元数 (w, x, y, z)
    """
    # 拆分四元数的实部和虚部
    w1, x1, y1, z1 = q_target.unbind(-1)
    w2, x2, y2, z2 = q_src.unbind(-1)
    
    # 计算乘积的实部和虚部，形状为 (N,)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # 将结果堆叠为 (N, 4) 的张量
    return torch.stack((w, x, y, z), dim=-1)


# pytorch3d code
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def rotmat_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Not differentiable!
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
# pytorch3d code

