from copy import deepcopy
import gc
import logging
printi, printd = logging.getLogger('flm-occ').info, logging.getLogger('flm-occ').debug

import torch
from torch import nn
import torch.nn.functional as F

from third_party.depth_anything_v2.dpt import DINOv2, DPTHead



def forward_nodepth(self, out_features, patch_h, patch_w):
    out = []
    for i, x in enumerate(out_features):
        if self.use_clstoken:
            x, cls_token = x[0], x[1]
            readout = cls_token.unsqueeze(1).expand_as(x)
            x = self.readout_projects[i](torch.cat((x, readout), -1))
        else:
            x = x[0]
        
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        
        x = self.projects[i](x)
        x = self.resize_layers[i](x)
        
        out.append(x)
    
    layer_1, layer_2, layer_3, layer_4 = out
    
    layer_1_rn = self.scratch.layer1_rn(layer_1)
    layer_2_rn = self.scratch.layer2_rn(layer_2)
    layer_3_rn = self.scratch.layer3_rn(layer_3)
    layer_4_rn = self.scratch.layer4_rn(layer_4)
    
    path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
    path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
    path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
    path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
    
    return {1: path_1, 2: path_2, 3: path_3, 4: path_4}


def forward(self, out_features, patch_h, patch_w):
    dict_paths = forward_nodepth(self, out_features, patch_h, patch_w)
    
    out = self.scratch.output_conv1(dict_paths[1])
    num_out = out.shape[0] * out.shape[1] * int(patch_h * 14) * int(patch_w * 14)
    if num_out > 1<<31:
        B = out.shape[0]
        assert num_out <= 1<<32, f'Too large feature map to process! {num_out} > 2x1^31.'
        out = torch.cat([
            F.interpolate(out[:B//2+1], (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True),
            F.interpolate(out[B//2+1:], (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        ], dim=0)
    else:
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        
    out = self.scratch.output_conv2(out)
        
    return out, dict_paths


class RGBEncoder(nn.Module):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    def __init__(self, 
            encoder,
            path_dav2,
            rank=-1,
        ):
        super().__init__()
        model_configs = self.model_configs
        self.rank = rank
        self.encoder = encoder
        features = model_configs[encoder]['features']
        self.features = features
        self.out_channels = model_configs[encoder]['out_channels']
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }[encoder]
        
        self.pretrained = DINOv2(model_name=encoder)
        self.path_ckpt = path_dav2
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, out_channels=self.out_channels)
        self.depth_head.forward = forward_nodepth.__get__(self.depth_head) # This would break `torch.jit.script` or `torch.compile`
        self.depth_head_ori = None


    def train(self, mode=True):
        super().train(mode)
        # self.pretrained.eval()
        # self.depth_head_ori.eval()
    
    
    def load_pretrained_weights(self,):
        location = f'cuda:{self.rank}' if self.rank >=0 else 'cpu'
        weights = torch.load(self.path_ckpt, weights_only=True, map_location=location)
        
        #  metric model ckpt requires rename
        try:
            printd('Loading DAv2 weights provided by EmbodiedOcc...')
            renamed_weights = { k[len('module.'):]: v for k, v in weights['model'].items() }
        except:
            printd('Loading official DAv2 weights...')
            renamed_weights = weights
        self.load_state_dict(renamed_weights)
        printd(f'Loaded DAv2 weights from {self.path_ckpt}')

        
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        
        del self.pretrained.mask_token
        
        return self


    def lora(self,):
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=8,                          # 低秩秩数
            lora_alpha=16,                # 缩放因子
            target_modules=['attn.qkv'],  # DINOv2 是 ViT 架构，注意力层包含 q, k, v
            lora_dropout=0.1,             # 可选 dropout
            bias="none",                  # 不训练偏置
        )
        
        self.pretrained = get_peft_model(self.pretrained, lora_config)
        
        self.pretrained.print_trainable_parameters()
        trainable_params, all_param = self.pretrained.get_nb_trainable_parameters()
        printd(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )
        
        return self


    def forward(self, x, h=480, w=640):
        """
        For 1xDPT and no depth lifter
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx, return_class_token=True)
        dict_paths = self.depth_head(features, patch_h, patch_w)
        cls_tokens = torch.stack([f[1] for f in features], dim=-2)  # [B, 4, 768]
        
        return None, dict_paths, cls_tokens
