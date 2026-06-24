import logging
printi, printd = logging.getLogger('flm-occ').info, logging.getLogger('flm-occ').debug

from typing import List, Optional, Union
from torch import nn



class SparseGaussianFormer(nn.Module):
    def __init__(
        self,
        gs_encoder,
        norm_layer,
        ffn,
        deformable_model,
        refine_layer,
        num_block: int = 6,
        attn = None,
        operation_order: Optional[List[str]] = None,
    ):
        super().__init__()
        self.gs_encoder = gs_encoder
        self.op_config_map = {
            'norm': norm_layer,
            'ffn': ffn,
            'deformable': deformable_model,
            'refine': refine_layer,
            'attn': attn,
            'identity': nn.Identity,
            'add': nn.Identity,
        }
        
        self.num_block = num_block
        if operation_order is None:
            operation_order = [
                "attn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_block
        self.operation_order = operation_order
        
        self.layers = nn.ModuleDict({
            f'{l}_{op}': self.op_config_map[op]() for l, op in enumerate(operation_order)
        })
        
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            k = f'{i}_{op}'
            if self.layers[k] is None:
                continue
            elif op != "refine":
                for p in self.layers[k].parameters():
                    if p.dim() > 1:
                        printd(f"init_weight for {k}")
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                printd(f"init_weight for {m.__class__.__name__}")
                m.init_weight()    
    
    
    def forward(self, BGS, BGSfeat, feature_maps, BK, BWH):
        BGSembed = self.gs_encoder(BGS) # [B, num_gs, dim_feat]
        prediction = []
        for i, op in enumerate(self.operation_order):
            # print(op)
            layer = self.layers[f'{i}_{op}']
            if op == 'attn':
                BGSfeat = layer(BGSfeat, BGS, BK, BWH)
            elif op == "norm" or op == "ffn":
                BGSfeat = layer(BGSfeat)
            elif op == "identity":
                identity = BGSfeat
            elif op == "add":
                BGSfeat = BGSfeat + identity
            elif op == "deformable":
                BGSfeat = layer(
                    BGSfeat,
                    BGS,
                    BGSembed,
                    feature_maps,
                    BK, BWH,
                )
            elif "refine" in op:
                BGS = layer(
                    BGSfeat,
                    BGS,
                    BGSembed,
                )
                prediction.append(BGS)
                
                if i != len(self.operation_order) - 1:
                    BGSembed = self.gs_encoder(BGS)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return prediction
