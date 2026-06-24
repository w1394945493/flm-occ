
import os; opj = os.path.join
import glob
import pickle
from typing import Literal

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets_.transform import DAv2_transform



def check_points_in_frustum2(points, K, WH, near, far):
    if isinstance(near, torch.Tensor):
        near = near[..., None]
    if isinstance(far, torch.Tensor):
        far = far[..., None]
    z = points[..., 2]
    mask = (z >= near) & (z <= far)  # filter points outside near and far planes
    points_ = points[mask]
    u = points_[..., 0] * K[..., 0, 0] / points_[..., 2] + K[..., 0, 2]
    v = points_[..., 1] * K[..., 1, 1] / points_[..., 2] + K[..., 1, 2]
    
    mask2 = torch.ones_like(mask)
    mask2[mask] = (u >= 0) & (u < WH[..., None, 0]) & (v >= 0) & (v < WH[..., None, 1])
    
    return mask & mask2

colormap = np.array([
    [ 22, 191, 206],
    [214,  38,  40],
    [ 43, 160,  43],
    [158, 216, 229],
    [114, 158, 206],
    [204, 204,  91],
    [255, 186, 119],
    [147, 102, 188],
    [ 30, 119, 181],
    [188, 188,  33],
    [255, 127,  12],
    [196, 175, 214],
    [153, 153, 153],
], dtype=np.uint8)

namemap = [
    'empty',
    'ceiling',
    'floor',
    'wall',
    'window',
    'chair',
    'bed',
    'sofa',
    'table',
    'tvs',
    'furniture',
    'objects',
    'unknown',
]

cls_freq = np.array([5080655412, 722756, 44793226, 41084591, 3416464, 21897101, 10609339,
          13846320, 23470172, 263393, 30949122, 9871618, 3196722886])

class ScannetDataset(Dataset):
    n_classes = 12
    namemap = namemap
    colormap = colormap
    vox_size = 0.08 # meters
    voxel_size = vox_size
    def __init__(
        self,
        config,
        split: Literal['train', 'val', 'test']='train',
        global_rank: int=0,
    ):

        self.occscannet_root = config.dir_dataset
        # data_tg='base' # config.base_or_mini
        data_tg=getattr(config, 'data_tg', 'base') # config.base_or_mini

        self.num_semantics = config.num_semantics
        assert self.n_classes == self.num_semantics, f"{self.n_classes} v.s. {self.num_semantics}"
        self.include_empty = config.include_empty # for labels_nempty
        self.cls_weights = 1 / torch.from_numpy(
            cls_freq[:12] if self.include_empty else cls_freq[1:12]).float().log()

        self.split = 'test' if split == 'val' else split

        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        if data_tg == 'base':
            subscenes_list = f'{self.occscannet_root}/{self.split}_final.txt'
        elif data_tg == 'mini':
            subscenes_list = f'{self.occscannet_root}/{self.split}_mini_final.txt'
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                self.used_subscenes[i] = f'{self.occscannet_root}/' + self.used_subscenes[i].strip()
        self.used_subscenes = self.used_subscenes[:100] if config.debug else self.used_subscenes
        
        self.grid_size = [60, 60, 36]
        voxgrid_scene = np.stack(np.where(np.ones(self.grid_size, dtype=bool)), axis=1)
        self.voxgrid_scene = torch.from_numpy(voxgrid_scene).float()

        self.random_offset = config.random_offset
        
        # 2d img
        color_jitter = config.color_jitter
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.fliplr = config.fliplr if split == 'train' else 0.0
        self.transform = DAv2_transform
        self.depth_min_max = [config.depth_min, config.depth_max]
        
        self.global_rank = global_rank


    def __len__(self):
        return len(self.used_subscenes)


    def __getitem__(self, idx):
        item = self._getitem(idx)
        while item[9].numel() == 0:
            print(self.global_rank, f"Warning: {item[-5]['this_name']} has no non-empty voxels, resampling...")
            item = self._getitem(torch.randint(len(self.used_subscenes), ()).item())
        return item


    def _getitem(self, idx):
        name = self.used_subscenes[idx]
        with open(name, 'rb') as f:
            data = pickle.load(f)
        
        name_without_ext = os.path.splitext(name)[0]
        this_name = name_without_ext.split('gathered_data/')[-1]
        
        rgb_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.jpg'
        depth_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.png'

        # region Image
        ## load
        img = Image.open(rgb_path).convert("RGB")
        WH_rgb0 = torch.tensor([img.size[0], img.size[1]])
        # resize
        img = img.resize((640, 480), Image.LANCZOS) # consistent with NYUv2
        ## Augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        ## Normalize
        rgb = self.transform(img)
        WH_rgb = torch.tensor([rgb.shape[2], rgb.shape[1]])
        K_rgb = torch.from_numpy(data['intrinsic'][:3,:3]).clone() # copy to avoid modifying the original one
        K_rgb[:2] *= (WH_rgb / WH_rgb0).view(2, 1)
        
        # region Depth
        depth = np.array(Image.open(depth_path).convert('I;16'), dtype=np.float32) / 1000 # meters
        depth = torch.from_numpy(depth)
        
        # region Coordinate
        pose = torch.from_numpy(data["cam_pose"].copy()).float()
        extrinsics = pose.inverse()
        # world(NYU)/lidar(SemKITTI) cooridnates of the voxel at voxel (0, 0, 0)
        vox_origin = torch.tensor(data["voxel_origin"]).float()
        
        # region Voxel grid
        dense_voxgrid = data[f"target_1_4"] # 60, 60, 36
        dense_voxgrid = np.moveaxis(dense_voxgrid, 0, 1)
        dense_voxgrid[dense_voxgrid==255] = 0
        mask_nempty = (0 < dense_voxgrid) & (dense_voxgrid < self.n_classes)
        
        # region Point coud
        vox_nempty = self.voxgrid_scene[mask_nempty.flatten()]
        offset_random = torch.rand_like(vox_nempty) if self.split=='train' and self.random_offset else 0.5
        pts_nempty_world = vox_origin + (vox_nempty + offset_random) * self.vox_size  # [N, 3]
        pts_nemtpy_cam = pts_nempty_world @ extrinsics[:3, :3].T + extrinsics[:3, 3]  # [N, 3]
        labels_nempty = torch.from_numpy(dense_voxgrid[mask_nempty].copy()).long()
        
        pts_scene = vox_origin + (self.voxgrid_scene + 0.5) * self.vox_size
        pts_scene_cam = pts_scene @ extrinsics[:3, :3].T + extrinsics[:3, 3]  # [N, 3]
        mask_frustum = check_points_in_frustum2(
            pts_scene_cam, K_rgb, WH_rgb, self.depth_min_max[0], self.depth_min_max[1]
        )
        # dummy mask for debug
        # mask_frustum = torch.ones(pts_scene_cam.shape[0], dtype=bool)
        pts_full_cam = pts_scene_cam[mask_frustum]  # [M, 3]
        labels_full = torch.from_numpy(dense_voxgrid.flatten()[mask_frustum].copy()).long()

        # region Flip
        flag_flip = False
        if np.random.rand() < self.fliplr:
            flag_flip = True
            rgb = torch.flip(rgb, dims=[-1])
            K_rgb[0, 2] = WH_rgb[0] - K_rgb[0, 2] - 1
            depth = torch.flip(depth, dims=[-1])

            pose[0] *= -1
            pose[:,0] *= -1
            
            pts_full_cam[:, 0] *= -1
            pts_nemtpy_cam[:, 0] *= -1
            
        # voxel_size for aggregation
        vsize_agg = self.vox_size * 4
        
        if not self.include_empty:
            labels_full = labels_full - 1
            labels_nempty = labels_nempty - 1
        
        metas = {
            'name': name,
            'this_name': this_name,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
        }
        
        # infos for EmbodiedOcc version of scene initialization
        world_near = vox_origin
        world_far = world_near + torch.tensor(self.scene_size)
        if flag_flip:
            world_near[0], world_far[0] = - world_far[0], - world_near[0]
        
        return (
            idx, rgb, K_rgb, WH_rgb, depth, pose, 
            pts_full_cam, labels_full,
            pts_nemtpy_cam, labels_nempty,
            vsize_agg, self.cls_weights,
            metas, # for debug
        )


    @staticmethod
    def collate_fn(data):
        (
            list_idx, rgbs, Ks, WHs, depths, poses,
            list_XYZ_query, list_labels_query,
            list_xyz_nempty, list_labels_nempty,
            list_voxsize_agg, list_cls_weights,
            list_metas,
        ) = zip(*data)
        rgbs = torch.stack(rgbs).float()
        depths = torch.stack(depths).float()
        Ks = torch.stack(Ks).float()
        WHs = torch.stack(WHs).float()
        poses = torch.stack(poses).float()
        cls_weights = torch.stack(list_cls_weights).float()
        
        data = {
            'idxs': list_idx,

            'rgbs': rgbs,
            'depths': depths,
            'Ks': Ks,
            'WHs': WHs, # be careful of the order
            'poses': poses,  # transform from camera to world
            'extrinsics': poses.inverse(),
            
            # positions that require supervision
            'XYZ_query': list_XYZ_query,
            'labels_query': list_labels_query,
            'xyz_nempty': list_xyz_nempty,
            'labels_nempty': list_labels_nempty,
            'voxsize_agg': list_voxsize_agg,  # for aggregation function
            'cls_weights': cls_weights, # for loss weighting
            
            # for debug
            'metas': list_metas,
        }

        return data
