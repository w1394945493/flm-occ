
import torch
from torch import nn

from utils.math import safe_inverse_sigmoid, safe_sigmoid, rotation_from_quaternion



scale_affine = None
e1e2_affine = None

class Superquadrics(nn.Module):
    # For Occ-ScanNet
    def __init__(self,
            num_gs,
            num_geometrics=13,
            num_semantics=0,
            learnable_init=True,
        ):
        super().__init__()
        
        self.num_gs = num_gs
        self.num_geometrics = num_geometrics
        self.num_semantics = num_semantics
        self.learnable_init = learnable_init
        
        GS = torch.empty(num_gs, num_geometrics + num_semantics, dtype=torch.float32)
        
        # mean
        GS[:,:2] = safe_inverse_sigmoid(torch.rand(num_gs, 2, dtype=torch.float32))
        GS[:, 2] = torch.log(
            torch.rand(num_gs, dtype=torch.float32) * (5.5 - 1.0) + 1.0
        )
        
        # scale
        GS[:, 3:6] = torch.log(torch.ones_like(GS[:, :3]) * 0.08)
        
        # quaternion
        # Be careful, whether from kernel to camera or reverse is defined by deformable_layer and GMM/SQMM
        # In the codes of GaussianFormer and its follow-ups, it is from camera to local
        # We prefer local to camera, in order to represent all kernels in the camera frame
        GS[:, 6:10] = torch.zeros(num_gs, 4, dtype=torch.float32)
        GS[:, 6] = 1

        # opacity
        GS[:, 10] = safe_inverse_sigmoid(0.5 * torch.ones(num_gs, dtype=torch.float32))

        # e1, e2
        GS[:, 11:13] = safe_inverse_sigmoid(torch.ones(num_gs, 2, dtype=torch.float32) * 0.9 / 1.9)
        
        # semantics
        GS[:, num_geometrics:] = torch.randn(num_gs, num_semantics, dtype=torch.float32)
        
        self.GS_init = GS
        self.GS = nn.Parameter(GS.detach().clone(), requires_grad=learnable_init)  # fixed anchors
    
    
    def init_weight(self):
        self.GS.data = self.GS_init.detach().clone()


    def contracted_to_euclidean(BUVW, BK, BWH):
        BUV = safe_sigmoid(BUVW[..., :2])
        BZ = BUVW[..., 2].exp()
        
        BU = BUV[..., 0] * BWH[..., None, 0]
        BV = BUV[..., 1] * BWH[..., None, 1]
        BX = BZ * (BU - BK[..., None, 0, 2]) / BK[..., None, 0, 0]
        BY = BZ * (BV - BK[..., None, 1, 2]) / BK[..., None, 1, 1]
        
        BXYZ = torch.stack([BX, BY, BZ], dim=-1)
        
        return BXYZ
    
    
    def scale_transform(S):
        return scale_affine(S.exp())
    
    
    def space_transform(self, GS, K, WH):
        """_summary_

        Args:
            P  (Tensor): [..., num_geo+num_sem]
            K  (Tensor): [..., 3, 3]
            WH (Tensor): [..., 2]
        """
        # means
        m = Superquadrics.contracted_to_euclidean(GS[..., :3], K, WH)
        s = Superquadrics.scale_transform(GS[..., 3:6])
        R = rotation_from_quaternion(GS[..., 6:10]) # F.normalize inside get_rotation_matrix
        
        opa, e1e2 = safe_sigmoid(GS[..., 10:13]).split([1, 2], dim=-1)
        e1e2 = e1e2_affine(e1e2)
        
        # semantics
        sem = GS[..., self.num_geometrics:] if self.num_semantics > 0 else None
        
        return m, s, R, opa, e1e2, sem
