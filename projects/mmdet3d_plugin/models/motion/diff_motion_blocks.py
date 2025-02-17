from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.runner.base_module import Sequential
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk
from diffusers.schedulers import DDIMScheduler
try:
    from projects.mmdet3d_plugin.ops import deformable_aggregation_function as DAF
except:
    DAF = None
from projects.mmdet3d_plugin.models.motion.modules.conditional_unet1d import ConditionalUnet1D, SinusoidalPosEmb
import torch.nn.functional as F


@PLUGIN_LAYERS.register_module()
class ModulationLayer(BaseModule):

    def __init__(
        self,
        embed_dims: int = 256,
        if_global_cond: bool = False,
        if_zeroinit_scale: bool = True,
    ):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale = if_zeroinit_scale
        self.if_global_cond = if_global_cond
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dims, embed_dims * 2) if not if_global_cond else
            nn.Linear(embed_dims * 2, embed_dims * 2),
            # Rearrange('batch t -> batch t 1'),
        )

    def init_weight(self):
        # Zero initialize the last layer of scale_shift_mlp
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], axis=-1)
        else:
            global_feature = time_embed
        scale_shift = self.scale_shift_mlp(global_feature)
        # scale_shift = torch.repeat_interleave(scale_shift,repeats=bs,dim=0)
        scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature


@PLUGIN_LAYERS.register_module()
class V1ModulationLayer(BaseModule):

    def __init__(
        self,
        embed_dims: int = 256,
        if_global_cond: bool = False, # False
        if_zeroinit_scale: bool = True, # False
    ):
        super(V1ModulationLayer, self).__init__()
        self.if_zeroinit_scale = if_zeroinit_scale
        self.if_global_cond = if_global_cond
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dims, embed_dims * 2) if not if_global_cond else
            nn.Linear(embed_dims * 2, embed_dims * 2),
            # Rearrange('batch t -> batch t 1'),
        )

    def init_weight(self):
        # Zero initialize the last layer of scale_shift_mlp
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
    ):
        """ 이를 통해 모델은 시간에 따른 변화에 traj_feature 가 유연하게 대응할 수 있게 됩니다.
        traj_feature: (b, 6, 256)
        time_embed: (b, 6, 256)
        global_cond = None
        """
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], axis=-1)
        else:
            global_feature = time_embed
        # import ipdb;ipdb.set_trace()
        # scale_shift: (b, 6, 512)
        scale_shift = self.scale_shift_mlp(global_feature)
        # scale_shift = torch.repeat_interleave(scale_shift,repeats=bs,dim=0)
        # scale_shift = scale_shift.unsqueeze(1)
        # scale, shift = (b, 6, 256), (b, 6, 256)
        scale, shift = scale_shift.chunk(2, dim=-1)
        # traj_feature: (b, 6, 256)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature


class TrajSparsePoint3DKeyPointsGenerator(BaseModule):

    def __init__(
            self,
            embed_dims: int = 256,
            num_sample: int = 20,
            num_learnable_pts: int = 0,
            fix_height: Tuple = (0,),
            ground_height: int = 0,
    ):
        super(TrajSparsePoint3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        # if self.num_learnable_pts > 0:
        #     self.learnable_fc = Linear(self.embed_dims, self.num_pts * 2)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height

    # def init_weight(self):
    #     if self.num_learnable_pts > 0:
    #         xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        """ 2D 점들에 고정된 높이(fixed height)와 지면 높이(ground height)를 더해 3D 좌표로 확장하는 것
        anchor: plan_reg_cum: (b, modal_num(=6), ego_fut_ts(=6) * 2)
            ego vehicle future trajectory (modal_num이 있고, 각 modal마다 ego_fut_ts * 2개의 2D point)

        return
            key_points: (b, modal_num(=6), ego_fut_ts(=6) * 5, 3)
        """
        # import ipdb; ipdb.set_trace()
        # assert self.num_learnable_pts > 0, 'No learnable pts'
        bs, num_anchor, _ = anchor.shape
        # key_points: (b, modal_num(=6), ego_fut_ts(=6), 2)
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        # offset: (b, modal_num(=6), ego_fut_ts(=6), 5, 1, 2)
        offset = torch.zeros(
            [bs, num_anchor, self.num_sample,
             len(self.fix_height), 1, 2],
            device=anchor.device,
            dtype=anchor.dtype)
        """
        offset: (b, modal_num(=6), ego_fut_ts(=6), len(fix_height) = 5, 1, 2)
        key_points[..., None, None, :] : (b, modal_num(=6), ego_fut_ts(=6), 1, 1, 2)
        key_points -> (b, modal_num(=6), ego_fut_ts(=6), 5, 1, 2)
        """
        key_points = offset + key_points[..., None, None, :]
        """ 3차원 좌표로 확장: ground_height 추가
        key_points -> (b, modal_num(=6), ego_fut_ts(=6), 5, 1, 3)
        """
        key_points = torch.cat(
            [
                key_points,
                key_points.new_full(key_points.shape[:-1] + (1,),
                                    fill_value=self.ground_height),
            ],
            dim=-1,
        )
        """ fix_height: (len(fix_height)= 5,) 
        height_offset: (len(fix_height)= 5, 2) # zeros
        """
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        """
        height_offset -> (len(fix_height)= 5, 3)
        0, 0, 0
        0, 0, 0.5
        0, 0, -0.5
        0, 0, 1
        0, 0, -1
        """
        height_offset = torch.cat([height_offset, fix_height[:, None]], dim=-1)
        """
        height_offset[None, None, None, :, None] -> (1, 1, 1, 5, 1, 3)
        key_points -> (b, modal_num(=6), ego_fut_ts(=6), 5, 1, 3)
        """
        key_points = key_points + height_offset[None, None, None, :, None]
        """
        key_points -> (b, modal_num(=6), ego_fut_ts(=6), 5, 1, 3) -> (b, modal_num(=6), ego_fut_ts(=6) * 5, 3)
        """
        key_points = key_points.flatten(2, 4)
        if (cur_timestamp is None or temp_timestamps is None or
                T_cur2temp_list is None or len(temp_timestamps) == 0):
            return key_points

        temp_key_points_list = []
        for i, t_time in enumerate(temp_timestamps):
            temp_key_points = key_points
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (T_cur2temp[:, None, None, :3] @ torch.cat(
                [
                    temp_key_points,
                    torch.ones_like(temp_key_points[..., :1]),
                ],
                dim=-1,
            ).unsqueeze(-1))
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    # @staticmethod
    def anchor_projection(
        self,
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            dst_anchor = anchor.clone()
            bs, num_anchor, _ = anchor.shape
            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample,
                                            -1).flatten(1, 2)
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1)

            dst_anchor = (torch.matmul(T_src2dst[..., :2, :2],
                                       dst_anchor[..., None]).squeeze(dim=-1) +
                          T_src2dst[..., :2, 3])

            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample,
                                            -1).flatten(2, 3)
            dst_anchors.append(dst_anchor)
        return dst_anchors


@PLUGIN_LAYERS.register_module()
class DiffMotionPlanningRefinementModule(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()

        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dims),
            nn.Linear(embed_dims, embed_dims * 4),
            nn.Mish(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dims, embed_dims * 2),
            # Rearrange('batch t -> batch t 1'),
        )
        # self.plan_cls_branch = nn.Sequential(
        #     *linear_relu_ln(embed_dims, 1, 2),
        #     Linear(embed_dims, 1),
        # )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    # def init_weight(self):
    #     # nn.init.normal_(self.time_mlp[1].weight, std=0.02)
    #     # nn.init.normal_(self.time_mlp[3].weight, std=0.02)
    #     nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
    #     nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)
    #     nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
    #     nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        timesteps,
    ):
        bs = traj_feature.shape[0]
        # 5. embed timestep
        time_embed = self.time_mlp(timesteps)
        scale_shift = self.scale_shift_mlp(time_embed)
        # scale_shift = torch.repeat_interleave(scale_shift,repeats=bs,dim=0)
        scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift

        # 6. get final prediction
        traj_delta = self.plan_reg_branch(traj_feature)
        reconstructed_traj = traj_delta.view(bs, self.ego_fut_ts, 2)
        plan_reg = reconstructed_traj

        return plan_reg


@PLUGIN_LAYERS.register_module()
class V0P1DiffMotionPlanningRefinementModule(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(V0P1DiffMotionPlanningRefinementModule, self).__init__()

        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dims),
            nn.Linear(embed_dims, embed_dims * 4),
            nn.Mish(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dims, embed_dims * 2, bias=True),
            # Rearrange('batch t -> batch t 1'),
        )
        # self.plan_cls_branch = nn.Sequential(
        #     *linear_relu_ln(embed_dims, 1, 2),
        #     Linear(embed_dims, 1),
        # )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    def init_weight(self):
        # Zero initialize the last layer of scale_shift_mlp
        nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
        nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_mlp[1].weight, std=0.02)
        nn.init.normal_(self.time_mlp[3].weight, std=0.02)
        # import ipdb; ipdb.set_trace()

    def forward(
        self,
        traj_feature,
        timesteps,
    ):
        bs = traj_feature.shape[0]
        # 5. embed timestep
        time_embed = self.time_mlp(timesteps)
        # import ipdb; ipdb.set_trace()
        scale_shift = self.scale_shift_mlp(time_embed)
        # scale_shift = torch.repeat_interleave(scale_shift,repeats=bs,dim=0)
        scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift

        # 6. get final prediction
        traj_delta = self.plan_reg_branch(traj_feature)
        reconstructed_traj = traj_delta.view(bs, self.ego_fut_ts, 2)
        plan_reg = reconstructed_traj

        return plan_reg


@PLUGIN_LAYERS.register_module()
class TrajPooler(BaseModule):

    def __init__(self, embed_dims=256, ego_fut_ts=6):
        super(TrajPooler, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.kps_generator = TrajSparsePoint3DKeyPointsGenerator(
            embed_dims=embed_dims,
            num_sample=ego_fut_ts,
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023,
        )

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(projection_mat[:, :, None, None],
                                 pts_extend[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3],
                                                     min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    def pool_feature_from_traj(self, noisy_traj_points, metas, feature_maps):
        modal_num = 1
        bs, _, _ = noisy_traj_points.shape
        plan_reg_cum = noisy_traj_points.view(bs, modal_num,
                                              self.ego_fut_ts * 2)
        # bs, modal_num, self.ego_fut_ts*5, 3
        key_points = self.kps_generator(plan_reg_cum)
        one_weights = torch.ones(
            [bs, modal_num, 6 * 4 * self.ego_fut_ts * 5, 8])
        one_weights = one_weights.to(device=noisy_traj_points.device,
                                     dtype=noisy_traj_points.dtype)
        weights = one_weights.softmax(dim=-2).reshape(
            bs,
            modal_num,
            6,  #self.num_cams,
            4,  #self.num_levels,
            self.ego_fut_ts * 5,  #self.num_pts,
            8,  #self.num_groups,
        )
        # attn_drop = 0.15
        attn_drop = 0.
        if self.training and attn_drop > 0:
            mask = torch.rand(bs, modal_num, 6, 1, self.ego_fut_ts * 5, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > attn_drop) * weights) / (1 - attn_drop)
        points_2d = (self.project_points(
            key_points,
            metas["projection_mat"],
            metas.get("image_wh"),
        ).permute(0, 2, 3, 1, 4).reshape(bs, modal_num, self.ego_fut_ts * 5, 6,
                                         2))
        weights = (weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
            bs,
            modal_num,
            self.ego_fut_ts * 5,
            6,
            4,
            8,
        ))
        # import ipdb;ipdb.set_trace()
        features = DAF(*feature_maps, points_2d,
                       weights).reshape(bs, modal_num, self.embed_dims)
        return features

    def forward(self, trajs, metas, feature_maps):

        trajs_cum = trajs.cumsum(dim=-2)
        traj_feature = self.pool_feature_from_traj(trajs_cum, metas,
                                                   feature_maps)
        return traj_feature


@PLUGIN_LAYERS.register_module()
class V1TrajPooler(BaseModule):

    def __init__(self, embed_dims=256, ego_fut_ts=6):
        super(V1TrajPooler, self).__init__()
        self.embed_dims = embed_dims  # 256
        self.ego_fut_ts = ego_fut_ts
        self.num_cams = 6
        self.num_levels = 4
        self.num_groups = 8
        self.num_pts = self.ego_fut_ts * 5
        self.attn_drop = 0.
        self.kps_generator = TrajSparsePoint3DKeyPointsGenerator(
            embed_dims=embed_dims,  # 256
            num_sample=ego_fut_ts,  # 6
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023,
        )
        self.proj_drop = nn.Dropout(0.0)
        self.residual_mode = "add"
        use_camera_embed = True
        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12))
            self.weights_fc = Linear(
                embed_dims, self.num_groups * self.num_levels * self.num_pts)
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, self.num_groups * self.num_cams * self.num_levels *
                self.num_pts)
        self.output_proj = Linear(embed_dims, embed_dims)

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        """생성된 3D 키포인트를, 메타 정보에 있는 카메라 프로젝션 행렬을 사용해 2D 이미지 평면에 투영합니다.


        """
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(projection_mat[:, :, None, None],
                                 pts_extend[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3],
                                                     min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    def _get_weights(self, instance_feature, metas=None):
        """

        instance_feature = traj_feature: (b, 6, 256)
        metas['projection_mat']: (b, 6, 4, 4)
        """
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature # (b, 6, 256)
        if self.camera_encoder is not None:
            a = metas["projection_mat"][:, :, :3] # (b, 6, 3, 4)
            b = a.reshape(bs, self.num_cams, -1) # (b, 6, 12)
            # camera_embed: (b, 6, 256)
            camera_embed = self.camera_encoder(b)
            feature = feature[:, :, None] + camera_embed[:, None]
        # feature: (b, 6, 6, 256)
        weights = self.weights_fc(feature)
        # weights: (b, 6, 6, 960) -> (b, 6, 720, 8)
        weights = weights.reshape(
            bs, num_anchor, -1, self.num_groups)
        # weights: (b, 6, 720, 8) -> (b, 6, 720, 8)
        weights = weights.softmax(dim=-2)
        # weights: (b, 6, 720, 8) -> (b, 6, num_cams=6, num_levels=4, num_pts=30, num_groups=8)
        weights = weights.reshape(
                bs,
                num_anchor,
                self.num_cams, #6
                self.num_levels, # 4
                self.num_pts, # 30
                self.num_groups,
            )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
        return weights

    def pool_feature_from_traj(self,
                               instance_feature,
                               noisy_traj_points,
                               metas,
                               feature_maps,
                               modal_num=1):
        """ projects/mmdet3d_plugin/models/motion/diff_motion_blocks.py
        instance_feature = traj_feature: (b, 6, 256)
        noisy_traj_points = trajs_cum: (b*6, 6, 2)
            - cum from diff_plan_reg
        feature_maps : List[Tensor]
                [0]: (1, 89760, 256)
                [1]: (6, 4, 2)
                [2]: (6, 4)
        modal_num = self.ego_fut_mode = 6
        """
        # modal_num = 1
        bs_modal, _, _ = noisy_traj_points.shape
        bs = bs_modal // modal_num
        # plan_reg_cum: (b, modal_num(=6), ego_fut_ts(=6) * 2)
        plan_reg_cum = noisy_traj_points.view(bs, modal_num,
                                              self.ego_fut_ts * 2)
        """
        plan_reg_cum: (b, modal_num(=6), ego_fut_ts(=6) * 2)
        key_points: (b, modal_num(=6), ego_fut_ts(=6) * 5, 3)
            - 각 점에 z 값을 추가했는데, 5개 높이에 대해 추가했다.
        """
        key_points = self.kps_generator(plan_reg_cum)
        # instance_feature = traj_feature: (b, 6, 256)
        # weights: (b, num_anchor=6, num_cams=6, num_levels=4, num_pts=30, num_groups=8)
        weights = self._get_weights(instance_feature, metas)
        """
        projection_mat: 카메라 프로젝션 행렬
        image_wh: 이미지의 너비와 높이
        """
        projection_mat = metas["projection_mat"] # (b, 6, 4, 4)
        image_wh = metas.get("image_wh") # (b, 6, 2)
        # points_2d: (b, 6, 6, 30, 2)
        """생성된 3D 키포인트를, 메타 정보에 있는 카메라 프로젝션 행렬을 사용해 2D 이미지 평면에 투영"""
        points_2d = self.project_points(
            key_points,
            projection_mat,
            image_wh,
        )
        # points_2d: (b, 6, 6, 30, 2) -> (b, num_cams=6, num_pts=30, num_anchor=6, 2)
        points_2d =  points_2d.permute(0, 2, 3, 1, 4)
        # points_2d: (b, 6, 30, 6, 2) -> (b, num_cams=6, num_pts=30, num_anchor=6, 2)
        points_2d = points_2d.reshape(bs, modal_num, self.num_pts, self.num_cams, 2)
        # weights: (b, num_anchor=6, num_cams=6, num_levels=4, num_pts=30, num_groups=8)
        # -> [1, num_anchor=6, num_pts=30, num_cams=6, num_levels=4, num_groups=8]
        weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous()
        # weights: [1, 6, 30, 6, 4, 8] -> [1, 6, 30, 6, 4, 8]
        weights = weights.reshape(
            bs,
            modal_num,
            self.ego_fut_ts * 5,
            6,
            4,
            8,
        )
        # import ipdb;ipdb.set_trace()
        """
이 함수는, 주어진 sampling 위치(투영된 2D 점들)에서 각 scale의 feature를 샘플링하고,
weight에 따라 가중치 합산을 수행해 aggregated feature를 생성합니다.

        feature_maps : List[Tensor]
                [0]: (1, 89760, 256)
                [1]: (6, 4, 2)
                [2]: (6, 4)
        points_2d: (b, num_cams=6, num_pts=30, num_anchor=6, 2)
        weights: [b, num_anchor=6, num_pts=30, num_cams=6, num_levels=4, num_groups=8]
        
        features: (b, 6, 256)
        """
        features = DAF(*feature_maps, points_2d,
                       weights)
        # features: (b, 6, 256) -> (b, 6, 256)
        features = features.reshape(bs, modal_num, self.embed_dims)
        # a: (b, 6, 256)
        a = self.output_proj(features)
        # output: (b, 6, 256)
        output = self.proj_drop(a)
        if self.residual_mode == "add":
            # output: (b, 6, 256)
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def forward(self,
                instance_feature,
                trajs,
                metas,
                feature_maps,
                modal_num=1):
        """
        목적
            noise 가 낀 trajectory points를, 카메라 프로젝션을 통해 이미지 평면에 투영하고,
            각 scale의 feature를 샘플링하고, weight에 따라 가중치 합산을 수행해 aggregated feature를 생성합니다.
            생성된 feature를 원래 instance feature에 더해줍니다.


        instance_feature = traj_feature: (b, 6, 256)
        trajs = diff_plan_reg: (b*6, 6, 2)
        feature_maps : List[Tensor]
                [0]: (1, 89760, 256)
                [1]: (6, 4, 2)
                [2]: (6, 4)
        modal_num = self.ego_fut_mode = 6
        """
        trajs_cum = trajs.cumsum(dim=-2)  # (b*6, 6, 2)
        # traj_feature: (b, 6, 256)
        traj_feature = self.pool_feature_from_traj(instance_feature,
                                                   trajs_cum,
                                                   metas,
                                                   feature_maps,
                                                   modal_num=modal_num)
        return traj_feature


@PLUGIN_LAYERS.register_module()
class V2TrajPooler(BaseModule):

    def __init__(self, embed_dims=256, ego_fut_ts=6):
        super(V2TrajPooler, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.kps_generator = TrajSparsePoint3DKeyPointsGenerator(
            embed_dims=embed_dims,
            num_sample=ego_fut_ts,
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023,
        )

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(projection_mat[:, :, None, None],
                                 pts_extend[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3],
                                                     min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    def pool_feature_from_traj(self,
                               noisy_traj_points,
                               metas,
                               feature_maps,
                               modal_num=1):

        # modal_num = 1
        bs, modal_num, _, _ = noisy_traj_points.shape
        # bs = bs_modal // modal_num
        plan_reg_cum = noisy_traj_points.view(bs, modal_num,
                                              self.ego_fut_ts * 2)
        # bs, modal_num, self.ego_fut_ts*5, 3
        key_points = self.kps_generator(plan_reg_cum)
        one_weights = torch.ones(
            [bs, modal_num, 6 * 4 * self.ego_fut_ts * 5, 8])
        one_weights = one_weights.to(device=noisy_traj_points.device,
                                     dtype=noisy_traj_points.dtype)
        weights = one_weights.softmax(dim=-2).reshape(
            bs,
            modal_num,
            6,  #self.num_cams,
            4,  #self.num_levels,
            self.ego_fut_ts * 5,  #self.num_pts,
            8,  #self.num_groups,
        )
        # attn_drop = 0.15
        attn_drop = 0.
        if self.training and attn_drop > 0:
            mask = torch.rand(bs, modal_num, 6, 1, self.ego_fut_ts * 5, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > attn_drop) * weights) / (1 - attn_drop)
        points_2d = (self.project_points(
            key_points,
            metas["projection_mat"],
            metas.get("image_wh"),
        ).permute(0, 2, 3, 1, 4).reshape(bs, modal_num, self.ego_fut_ts * 5, 6,
                                         2))
        weights = (weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
            bs,
            modal_num,
            self.ego_fut_ts * 5,
            6,
            4,
            8,
        ))
        # import ipdb;ipdb.set_trace()
        features = DAF(*feature_maps, points_2d,
                       weights).reshape(bs, modal_num, self.embed_dims)
        return features

    def forward(self, trajs, metas, feature_maps, modal_num=1):
        trajs_cum = trajs.cumsum(dim=-2)
        traj_feature = self.pool_feature_from_traj(trajs_cum,
                                                   metas,
                                                   feature_maps,
                                                   modal_num=modal_num)
        return traj_feature


@PLUGIN_LAYERS.register_module()
class V3TrajPooler(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=6,
        use_camera_embed=True,
        proj_drop=0.0,
        num_cams=6,
        num_levels=4,
        num_groups=8,
        attn_drop=0.,
        residual_mode="add",
    ):
        super(V3TrajPooler, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_pts = self.ego_fut_ts * 5
        self.attn_drop = attn_drop
        self.kps_generator = TrajSparsePoint3DKeyPointsGenerator(
            embed_dims=embed_dims,
            num_sample=ego_fut_ts,
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023,
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.residual_mode = residual_mode
        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12))
            self.weights_fc = Linear(embed_dims,
                                     num_groups * num_levels * self.num_pts)
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts)
        self.output_proj = Linear(embed_dims, embed_dims)

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(projection_mat[:, :, None, None],
                                 pts_extend[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3],
                                                     min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    def _get_weights(self, instance_feature, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1))
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (self.weights_fc(feature).reshape(
            bs, num_anchor, -1, self.num_groups).softmax(dim=-2).reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            ))
        if self.training and self.attn_drop > 0:
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
        return weights

    def pool_feature_from_traj(self,
                               instance_feature,
                               noisy_traj_points,
                               metas,
                               feature_maps,
                               modal_num=1):
        # modal_num = 1
        # import ipdb;ipdb.set_trace()c
        bs, modal_num = instance_feature.shape[:2]
        bs, modal_num, _, _ = noisy_traj_points.shape
        # bs = bs_modal // modal_num
        plan_reg_cum = noisy_traj_points.view(bs, modal_num,
                                              self.ego_fut_ts * 2)
        # bs, modal_num, self.ego_fut_ts*5, 3
        key_points = self.kps_generator(plan_reg_cum)

        weights = self._get_weights(instance_feature, metas)

        points_2d = (self.project_points(
            key_points,
            metas["projection_mat"],
            metas.get("image_wh"),
        ).permute(0, 2, 3, 1, 4).reshape(bs, modal_num, self.num_pts,
                                         self.num_cams, 2))
        weights = (weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
            bs,
            modal_num,
            self.num_pts,
            self.num_cams,
            self.num_levels,
            self.num_groups,
        ))
        # import ipdb;ipdb.set_trace()
        features = DAF(*feature_maps, points_2d,
                       weights).reshape(bs, modal_num, self.embed_dims)
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def forward(self,
                instance_feature,
                trajs,
                metas,
                feature_maps,
                modal_num=1):
        trajs_cum = trajs.cumsum(dim=-2)
        traj_feature = self.pool_feature_from_traj(instance_feature,
                                                   trajs_cum,
                                                   metas,
                                                   feature_maps,
                                                   modal_num=modal_num)
        return traj_feature


@PLUGIN_LAYERS.register_module()
class V1DiffMotionPlanningRefinementModule(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(V1DiffMotionPlanningRefinementModule, self).__init__()

        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dims),
            nn.Linear(embed_dims, embed_dims * 4),
            nn.Mish(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dims, embed_dims * 2),
            # Rearrange('batch t -> batch t 1'),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        # nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        traj_feature,
        timesteps,
    ):
        # import ipdb; ipdb.set_trace()
        bs = traj_feature.shape[0]

        # 5. embed timestep
        time_embed = self.time_mlp(timesteps)
        scale_shift = self.scale_shift_mlp(time_embed)
        # scale_shift = torch.repeat_interleave(scale_shift,repeats=bs,dim=0)
        scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, 1, self.ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        plan_cls = plan_cls.repeat(1, 3, 1).reshape(bs, 1, -1)
        # import ipdb; ipdb.set_trace()
        traj_delta = self.plan_reg_branch(traj_feature)
        # reconstructed_traj = traj_delta.view(bs,self.ego_fut_ts,2)
        plan_reg = traj_delta.reshape(bs, 1, self.ego_fut_mode, self.ego_fut_ts,
                                      2).repeat(1, 3, 1, 1, 1)
        plan_reg = plan_reg.view(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts,
                                 2)

        return plan_reg, plan_cls


@PLUGIN_LAYERS.register_module()
class V2DiffMotionPlanningRefinementModule(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(V2DiffMotionPlanningRefinementModule, self).__init__()

        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dims),
            nn.Linear(embed_dims, embed_dims * 4),
            nn.Mish(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dims * 2, embed_dims * 2),
            # Rearrange('batch t -> batch t 1'),
        )
        # self.plan_cls_branch = nn.Sequential(
        #     *linear_relu_ln(embed_dims, 1, 2),
        #     Linear(embed_dims, 1),
        # )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    def forward(
        self,
        traj_feature,
        timesteps,
        global_cond,
    ):
        bs = traj_feature.shape[0]
        # 5. embed timestep
        # import ipdb;ipdb.set_trace()
        time_embed = self.time_mlp(timesteps)
        global_feature = torch.cat([global_cond, time_embed], axis=-1)
        scale_shift = self.scale_shift_mlp(global_feature)
        # scale_shift = torch.repeat_interleave(scale_shift,repeats=bs,dim=0)
        scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift

        # 6. get final prediction
        traj_delta = self.plan_reg_branch(traj_feature)
        reconstructed_traj = traj_delta.view(bs, self.ego_fut_ts, 2)
        plan_reg = reconstructed_traj

        return plan_reg


@PLUGIN_LAYERS.register_module()
class V3DiffMotionPlanningRefinementModule(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(V3DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

    def init_weight(self):
        # import ipdb;ipdb.set_trace()
        nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
        nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

    def forward(
        self,
        traj_feature,
    ):
        bs = traj_feature.shape[0]

        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.view(bs, self.ego_fut_ts, 2)

        return plan_reg


@PLUGIN_LAYERS.register_module()
class V4DiffMotionPlanningRefinementModule(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        if_zeroinit_reg=True,
    ):
        super(V4DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.if_zeroinit_reg = if_zeroinit_reg

    def init_weight(self):
        # import ipdb;ipdb.set_trace()
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        # nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        traj_feature,
    ):
        """
        traj_feature: (b, 6, 256)

        return
            diff_plan_reg: (b, 1, 3*6, 6, 2)
            diff_plan_cls: (b, 1, 3*6)
        """
        bs = traj_feature.shape[0]
        # 6. get final prediction
        traj_feature = traj_feature.view(bs, 1, self.ego_fut_mode, -1) # (b, 1, 6, 256)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1) # plan_cls: (b, 1, 6, 1) -> (b, 1, 6)
        # plan_cls: (b, 1, 6) -> (b, 3, 6) -> (b, 1, 18)
        plan_cls = plan_cls.repeat(1, 3, 1).reshape(bs, 1, -1)

        # import ipdb; ipdb.set_trace()
        traj_delta = self.plan_reg_branch(traj_feature) # (b, 1, 6, 256) -> (b, 1, 6, 12)
        # reconstructed_traj = traj_delta.view(bs,self.ego_fut_ts,2)
        a = traj_delta.reshape(bs, 1, self.ego_fut_mode, self.ego_fut_ts,
                                      2) # a: (b, 1, 6, 6, 2)
        plan_reg = a.repeat(1, 3, 1, 1, 1) # plan_reg: (b, 3, 6, 6, 2)
        plan_reg = plan_reg.view(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts,
                                 2) # plan_reg: (b, 1, 18, 6, 2)
        return plan_reg, plan_cls


@PLUGIN_LAYERS.register_module()
class V5DiffMotionPlanningRefinementModule(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        if_zeroinit_reg=True,
    ):
        super(V5DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.if_zeroinit_reg = if_zeroinit_reg

    def init_weight(self):
        # import ipdb;ipdb.set_trace()
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        # nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        traj_feature,
    ):
        bs = traj_feature.shape[0]
        # import ipdb;ipdb.set_trace()
        # 6. get final prediction
        traj_feature = traj_feature.view(bs, self.ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        plan_cls = plan_cls.reshape(bs, -1)

        traj_delta = self.plan_reg_branch(traj_feature)

        # reconstructed_traj = traj_delta.view(bs,self.ego_fut_ts,2)
        plan_reg = traj_delta.reshape(bs, self.ego_fut_mode, self.ego_fut_ts, 2)
        # plan_reg = plan_reg.view(bs,1,3*self.ego_fut_mode,self.ego_fut_ts,2)

        return plan_reg, plan_cls


@PLUGIN_LAYERS.register_module()
class V6DiffMotionPlanningRefinementModule(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        if_zeroinit_reg=True,
    ):
        super(V6DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.if_zeroinit_reg = if_zeroinit_reg

    def init_weight(self):
        # import ipdb;ipdb.set_trace()
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        # nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        traj_feature,
        last_plan_reg,
    ):
        bs = traj_feature.shape[0]
        # import ipdb;ipdb.set_trace()
        # 6. get final prediction
        traj_feature = traj_feature.view(bs, self.ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        plan_cls = plan_cls.reshape(bs, -1)

        traj_delta = self.plan_reg_branch(traj_feature)

        # reconstructed_traj = traj_delta.view(bs,self.ego_fut_ts,2)
        plan_reg = traj_delta.reshape(bs, self.ego_fut_mode, self.ego_fut_ts, 2)
        pred_plan_reg = plan_reg + last_plan_reg
        # plan_reg = plan_reg.view(bs,1,3*self.ego_fut_mode,self.ego_fut_ts,2)

        return plan_reg, plan_cls
