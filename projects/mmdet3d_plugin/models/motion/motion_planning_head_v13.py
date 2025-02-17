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


class TrajSparsePoint3DKeyPointsGenerator(BaseModule):

    def __init__(
            self,
            embed_dims: int = 256,
            num_sample: int = 20,   # ego_fut_ts = 6
            num_learnable_pts: int = 0,
            fix_height: Tuple = (0,),  # (0, 0.5, -0.5, 1, -1)
            ground_height: int = 0,  # -1.84023
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
        # import ipdb; ipdb.set_trace()
        # assert self.num_learnable_pts > 0, 'No learnable pts'
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        offset = torch.zeros(
            [bs, num_anchor, self.num_sample,
             len(self.fix_height), 1, 2],
            device=anchor.device,
            dtype=anchor.dtype)

        key_points = offset + key_points[..., None, None, :]
        key_points = torch.cat(
            [
                key_points,
                key_points.new_full(key_points.shape[:-1] + (1,),
                                    fill_value=self.ground_height),
            ],
            dim=-1,
        )
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:, None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]
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


@HEADS.register_module()
class V13MotionPlanningHead(BaseModule):

    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,  # ego_fut_mode = 6
        if_init_timemlp=True,
        motion_anchor=None,
        plan_anchor=None,  # f'data/kmeans/kmeans_plan_{ego_fut_mode}.npy'
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        interact_operation_order=None,
        diff_operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        self_attn_model=None,
        mode_cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        diff_refine_layer=None,
        traj_pooler_layer=None,
        modulation_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
    ):
        super(V13MotionPlanningHead, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.interact_operation_order = interact_operation_order
        self.diff_operation_order = diff_operation_order
        self.if_init_timemlp = if_init_timemlp

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            # "mode_cross_gnn": [mode_cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
            "agent_cross_gnn": [graph_model, ATTENTION],
            "map_cross_gnn": [cross_graph_model, ATTENTION],
            "anchor_cross_gnn": [cross_graph_model, ATTENTION],
            "diff_refine": [diff_refine_layer, PLUGIN_LAYERS],
            "traj_pooler": [traj_pooler_layer, PLUGIN_LAYERS],
            "modulation": [modulation_layer, PLUGIN_LAYERS],
            "self_attn": [self_attn_model, ATTENTION],
        }
        # ERROR
        self.interact_layers = nn.ModuleList([
            build(*self.op_config_map.get(op, [None, None]))
            for op in self.interact_operation_order
        ])
        self.diff_layers = nn.ModuleList([
            build(*self.op_config_map.get(op, [None, None]))
            for op in self.diff_operation_order
        ])
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims,
                                       self.embed_dims * 2,
                                       bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2,
                                      self.embed_dims,
                                      bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)

        # motion init
        motion_anchor = np.load(motion_anchor)  # (10, 6, 12, 2)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )  # (10, 6, 12, 2)
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        plan_anchor = np.load(plan_anchor)  # (3, 6, 6, 2)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # # plan traj anchor init
        # self.plan_traj_pos_encoder = nn.Sequential(
        #     *linear_relu_ln(embed_dims, 1, 2,ego_fut_ts*2),
        # )
        self.plan_pos_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, 768),
            Linear(embed_dims, embed_dims),
        )
        self.num_det = num_det
        self.num_map = num_map

        self.kps_generator = TrajSparsePoint3DKeyPointsGenerator(
            embed_dims=embed_dims,
            num_sample=ego_fut_ts,
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023,
        )
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dims),
            nn.Linear(embed_dims, embed_dims * 4),
            nn.Mish(),
            nn.Linear(embed_dims * 4, embed_dims),
        )

    def init_weights(self):
        for i, op in enumerate(self.interact_operation_order):
            if self.interact_layers[i] is None:
                continue
            elif op != "refine":
                for p in self.interact_layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        # for i, op in enumerate(self.diff_operation_order):
        #     if self.diff_layers[i] is None:
        #         continue
        #     elif op != "diff_refine":
        #         for p in self.diff_layers[i].parameters():
        #             if p.dim() > 1:
        #                 nn.init.xavier_uniform_(p)
        # import ipdb;ipdb.set_trace()
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()
        # Initialize timestep embedding MLP:
        if self.if_init_timemlp:
            nn.init.normal_(self.time_mlp[1].weight, std=0.02)
            nn.init.normal_(self.time_mlp[3].weight, std=0.02)

    def get_motion_anchor(
        self,
        classification,  #
        prediction,
    ):
        """
        det_classification
            det_output['classification'][-1].shape: torch.Size([1, 900, 10])
        det_anchors
            det_output['prediction'][-1].shape: torch.Size([1, 900, 11])
        """
        cls_ids = classification.argmax(dim=-1)  # torch.Size([1, 900])
        # self.motion_anchor: (10, 6, 12, 2)
        # motion_anchor: (1, 900, 6, 12, 2)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()  # (1, 900, 11)
        # trajs_lidar: [1, 900, 6, 12, 2]
        trajs_lidar = self._agent2lidar(motion_anchor, prediction)
        return trajs_lidar

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack([
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ])

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        # import ipdb;ipdb.set_trace()
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(self.interact_layers[index](
            query,
            key,
            value,
            query_pos=query_pos,
            key_pos=key_pos,
            **kwargs,
        ))

    def diff_graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        # import ipdb;ipdb.set_trace()
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(self.diff_layers[index](
            query,
            key,
            value,
            query_pos=query_pos,
            key_pos=key_pos,
            **kwargs,
        ))

    def normalize_ego_fut_trajs(self, gt_ego_fut_trajs):
        """
주어진 Ego 차량의 미래 궤적 좌표(보통 x, y 좌표)를 정규화하여
    네트워크가 학습하기 적합한 범위로 스케일링
1. **좌표 분리 및 스케일링**
   - 입력으로 들어오는 `gt_ego_fut_trajs`의 첫 번째 차원은 x좌표, 두 번째 차원은 y좌표로 가정
   - x좌표는 3으로 나누어 줌으로써 스케일을 축소하고, 이후 `clamp(-1, 1)`을 적용하여 [-1, 1] 범위 내로 제한
   - y좌표는 먼저 0.5를 더한 뒤 8.1로 나눕니다.
        이로써 y좌표를 [0, 1] 범위로 압축한 후, 다시 `clamp(0, 1)`을 적용합니다.
        이후 y좌표에 대해 `*2 - 1` 연산을 수행하여 [-1, 1] 범위로 변환합니다.

2. **목적**
   - 이렇게 x와 y 좌표를 각각 [-1, 1] 범위로 정규화함으로써,
        네트워크 내의 후속 모듈(예를 들어, diffusion 단계나 회귀(branch) 등)에서
            입력으로 사용될 때 값의 범위가 일정해져 학습 안정성을 높이고,
        모델이 다양한 스케일의 데이터를 효과적으로 처리할 수 있도록 돕습니다.
   - 특히, 좌표값이 원래 단위나 범위가 서로 다를 수 있는데(예: x는 미터 단위로 3으로 나누고, y는 다른 스케일 적용),
        이 함수를 통해 모든 좌표를 동일한 범위로 맞춰줌으로써
        모델이 학습 과정에서 불필요한 scale 차이에 의한 영향을 받지 않도록 합니다.

3. **출력 스펙**
   - 최종 출력은 입력과 동일한 shape을 가지며,
        각 좌표값이 [-1, 1] 범위 내로 정규화된 tensor가 반환

        """
        # bs, ego_fut_ts, _ = gt_ego_fut_trajs.shape
        odo_info_fut_x = gt_ego_fut_trajs[..., 0:1]
        odo_info_fut_y = gt_ego_fut_trajs[..., 1:2]

        odo_info_fut_x = odo_info_fut_x / 3
        odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y + 0.5) / 8.1
        odo_info_fut_y = odo_info_fut_y.clamp(0, 1)
        odo_info_fut_y = odo_info_fut_y * 2 - 1
        odo_info_fut = torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        # odo_info_fut = odo_info_fut.reshape(-1,self.ego_fut_ts, 2)
        return odo_info_fut

    def denormalize_ego_fut_trajs(self, noisy_traj_points):
        """denormalize_ego_fut_trajs 함수는 정규화된 Ego 미래 궤적 좌표를 원래의 스케일로 변환하여,
        모델의 예측 결과를 실제 물리적 좌표 값으로 복원하는 전처리의 역과정
        """
        # bs, ego_fut_ts, _ = noisy_traj_points.shape
        odo_info_fut_x = noisy_traj_points[..., 0:1]
        odo_info_fut_y = noisy_traj_points[..., 1:2]

        odo_info_fut_x = odo_info_fut_x * 3
        # odo_info_fut_x = odo_info_fut_x.clamp(-1, 1)
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * 8.1 - 0.5
        # odo_info_fut_y = odo_info_fut_y.clamp(-1, 1)
        odo_info_fut = torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        return odo_info_fut

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

    def forward(
        self,
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):
        if self.training:
            return self.forward_train(
                det_output,
                map_output,
                feature_maps,
                metas,
                anchor_encoder,
                mask,
                anchor_handler,
            )
        else:
            return self.forward_test(
                det_output,
                map_output,
                feature_maps,
                metas,
                anchor_encoder,
                mask,
                anchor_handler,
            )

    def forward_train(
        self,
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected,
            anchor_embed_selected) = topk(det_confidence, self.num_det,
                                          instance_feature, anchor_embed)

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected,
            map_anchor_embed_selected) = topk(map_confidence, self.num_map,
                                              map_instance_feature,
                                              map_anchor_embed)

        # =========== get ego/temporal feature/anchor ===========
        # import ipdb;ipdb.set_trace()
        bs, num_anchor, dim = instance_feature.shape
        device = instance_feature.device
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)
        # import ipdb;ipdb.set_trace()
        # =========== mode anchor init ===========
        """motion_anchor: [1, 900, 6, 12, 2] ( 세계 좌표계 , 아니면 자차기준 좌표계)
        주변 900개 객체들의 움직임을 예측하기 위한 초기 anchor
            한 객체마다 (6, 12, 2) 의 초기 anchor
        """
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        """
        weight
            (3, 6, 6, 2) -> (1, 3, 6, 6, 2) -> (1, 3=cmd_mode, 6=modal_mode, 6=ego_fut_ts, 2)
        """
        plan_anchor = torch.tile(
            self.plan_anchor[None],
            (bs, 1, 1, 1, 1))  # bs, cmd_mode, modal_mode, ego_fut_ts, 2
        # =========== mode query init ===========

        motion_mode_query = self.motion_anchor_encoder(
            gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos)
        plan_mode_query = plan_mode_query.flatten(1, 2).unsqueeze(1)

        # # =========== plan query init ===========
        # gt_ego_fut_trajs = metas['gt_ego_fut_trajs']
        # odo_info_fut = self.normalize_ego_fut_trajs(gt_ego_fut_trajs)
        # timesteps = torch.randint(
        #     0, self.diffusion_scheduler.config.num_train_timesteps,
        #     (bs,), device=device
        # )# TODO, only bs timesteps
        # noise = torch.randn(odo_info_fut.shape, device=device)
        # noisy_traj_points = self.diffusion_scheduler.add_noise(
        #     original_samples=odo_info_fut,
        #     noise=noise,
        #     timesteps=timesteps,
        # ).float()
        # noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        # noisy_traj_points = self.denormalize_ego_fut_trajs(noisy_traj_points) # bs, ego_fut_ts, 2
        # noisy_traj_points_cum = noisy_traj_points.cumsum(dim=-2)
        # traj_feature = self.pool_feature_from_traj(noisy_traj_points_cum,metas,feature_maps) # bs,1,2
        # import ipdb;ipdb.set_trace()
        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat(
            [instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat(
            [anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)
        # TODO, need to add traj anchor embed into temp_anchor_embed & temp_instance_feature

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        # planning_diffusion_loss = []
        # import ipdb;ipdb.set_trace()
        for i, op in enumerate(self.interact_operation_order):
            if self.interact_layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(
                    bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.interact_layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.interact_layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                # import ipdb;ipdb.set_trace()
                motion_query = motion_mode_query + (
                    instance_feature +
                    anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed
                                               )[:, num_anchor:].unsqueeze(2)
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_reg,
                    plan_status,
                ) = self.interact_layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                    metas,
                )
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor],
                                         det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:],
                                           plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }

        # =================== diffusion part ====================

        # global_cond=(instance_feature + anchor_embed)[:, :num_anchor].mean(dim=1)
        # _, (instance_feature_selected, anchor_embed_selected) = topk(
        #     det_confidence, self.num_det, instance_feature[:,:num_anchor], anchor_embed[:,:num_anchor]
        # )
        # gt_ego_fut_trajs = metas['gt_ego_fut_trajs']
        # odo_info_fut = self.normalize_ego_fut_trajs(gt_ego_fut_trajs)
        # place_holder_num = self.ego_fut_mode - 1
        # odo_info_fut_placeholder = torch.randn(bs,place_holder_num,self.ego_fut_ts,2,device=device)/3
        # import ipdb;ipdb.set_trace()
        bs_indices = torch.arange(bs, device=plan_query.device)
        cmd = metas['gt_ego_fut_cmd'].argmax(dim=-1)
        # cmd_plan_nav_query = plan_nav_query[bs_indices, cmd]

        cmd_plan_anchor = plan_anchor[bs_indices, cmd]
        zeros_cat = torch.zeros(bs, 6, 1, 2, device=device)
        cmd_plan_anchor = torch.cat([zeros_cat, cmd_plan_anchor], dim=2)
        tgt_cmd_plan_anchor = cmd_plan_anchor[:, :,
                                              1:, :] - cmd_plan_anchor[:, :, :
                                                                       -1, :]
        odo_info_fut = self.normalize_ego_fut_trajs(tgt_cmd_plan_anchor)
        # import ipdb;ipdb.set_trace()
        # odo_info_fut = torch.cat([odo_info_fut.view(bs,1,self.ego_fut_ts,2), odo_info_fut_placeholder], dim=1)
        odo_info_fut = odo_info_fut.view(bs * self.ego_fut_mode,
                                         self.ego_fut_ts, 2)
        # odo_info_fut = odo_info_fut * self.noise_scale
        # TODO: multi mode, need to concat noise for other modes

        # import ipdb;ipdb.set_trace()
        # magic number 40 means that we add little noise for each anchor
        timesteps = torch.randint(0, 40, (bs,),
                                  device=device)  # TODO, only bs timesteps

        repeat_timesteps = timesteps.repeat_interleave(self.ego_fut_mode)
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=repeat_timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denormalize_ego_fut_trajs(
            noisy_traj_points)  # bs, ego_fut_ts, 2
        # noisy_traj_points_cum = noisy_traj_points.cumsum(dim=-2)
        # traj_feature = self.pool_feature_from_traj(noisy_traj_points_cum,metas,feature_maps) # bs,1,2
        # import ipdb;ipdb.set_trace()
        diff_plan_reg = noisy_traj_points
        traj_pos_embed = gen_sineembed_for_position(diff_plan_reg,
                                                    hidden_dim=128)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_pos_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, self.ego_fut_mode, -1)
        # traj_feature = traj_pos_feature
        # import ipdb;ipdb.set_trace()

        plan_nav_query = plan_query.squeeze(1)
        plan_nav_query = plan_nav_query.view(bs, 3, 6, -1)
        # bs_indices = torch.arange(bs, device=plan_query.device)
        # cmd = metas['gt_ego_fut_cmd'].argmax(dim=-1)
        cmd_plan_nav_query = plan_nav_query[bs_indices, cmd]

        time_embed = self.time_mlp(repeat_timesteps)
        time_embed = time_embed.view(bs, self.ego_fut_mode, -1)

        diff_planning_prediction = []
        diff_planning_classification = []
        repeat_ego_anchor_embed = ego_anchor_embed.repeat(
            1, self.ego_fut_mode, 1)
        # import ipdb;ipdb.set_trace()
        for i, op in enumerate(self.diff_operation_order):
            if self.diff_layers[i] is None:
                continue
            elif op == "traj_pooler":
                if len(diff_plan_reg.shape) != 3:
                    # import ipdb;ipdb.set_trace()
                    diff_plan_reg = diff_plan_reg[
                        :,
                        :,
                        -self.ego_fut_mode:,
                    ].flatten(0, 2)
                traj_feature = self.diff_layers[i](traj_feature,
                                                   diff_plan_reg,
                                                   metas,
                                                   feature_maps,
                                                   modal_num=self.ego_fut_mode)
            elif op == "self_attn":
                traj_feature = self.diff_layers[i](
                    traj_feature,
                    traj_feature,
                    traj_feature,
                )
            elif op == "modulation":
                # import ipdb;ipdb.set_trace()
                traj_feature = self.diff_layers[i](
                    traj_feature,
                    time_embed,
                    global_cond=global_cond
                    if self.diff_layers[i].if_global_cond else None,
                    # global_cond=
                )
            elif op == "agent_cross_gnn":
                # import ipdb;ipdb.set_trace()
                traj_feature = self.diff_graph_model(
                    i,
                    traj_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=repeat_ego_anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "map_cross_gnn":
                traj_feature = self.diff_layers[i](
                    traj_feature,
                    map_instance_feature_selected,
                    map_instance_feature_selected,
                    query_pos=repeat_ego_anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "anchor_cross_gnn":
                traj_feature = self.diff_layers[i](
                    traj_feature,
                    key=cmd_plan_nav_query,
                    value=cmd_plan_nav_query,
                )
            elif op == "norm" or op == "ffn":
                traj_feature = self.diff_layers[i](traj_feature)
            elif op == "diff_refine":
                # import ipdb;ipdb.set_trace()
                diff_plan_reg, diff_plan_cls = self.diff_layers[i](
                    traj_feature,)
                diff_planning_prediction.append(diff_plan_reg)
                diff_planning_classification.append(diff_plan_cls)
        planning_output["diffusion_prediction"] = diff_planning_prediction
        planning_output[
            "diffusion_classification"] = diff_planning_classification
        planning_output["tgt_cmd_plan_anchor"] = tgt_cmd_plan_anchor
        return motion_output, planning_output

    def forward_test(
            self,
            det_output,
            map_output,
            feature_maps,
            metas,
            anchor_encoder,  # SparseBox3DEncoder of det_head
            mask,  # InstanceBank.mask of det_head
            anchor_handler,  # InstanceBank.anchor_handler(=Sparse3DKeyPointsGenerator) of det_head
    ):
        ######### input ###############
        """ det_output : Dict
        instance_feature: (1, 900, 256)
        anchor_embed: (1, 900, 256)

        len(det_output['classification']): 6
            det_output['classification'][-1].shape: torch.Size([1, 900, 10])
        len(det_output['prediction']): 6
            det_output['prediction'][-1].shape: torch.Size([1, 900, 11])
        len(det_output['quality']): 6
            det_output['quality'][-1].shape: torch.Size([1, 900, 2])
        """
        """ map_output: Dict
        instance_feature: (1, 100, 256)
        anchor_embed: (1, 100, 256)
        len(map_output['classification']): 6
        map_output['classification'][-1].shape: torch.Size([1, 100, 3])
        len(map_output['prediction']): 6
        map_output['prediction'][-1].shape: torch.Size([1, 100, 40])
        """
        """ feature_maps : List[Tensor]
                [0]: (1, 89760, 256)
                [1]: (6, 4, 2)
                [2]: (6, 4)
        """
        ######### output ##############
        """ motion_output : Dict
        "classification": len = 1
            (1, 900, fut_mode=6)
        "prediction": len = 1
            (1, 900, fut_mode=6, fut_ts=12, 2)
        "period": (1, 900)
        "anchor_queue": len = 4
            (1, 900, 11)
        """
        """ planning_output : Dict
        classification: len 1
            (1, 1, cmd_mode(3)*modal_mode(6)=18)
        prediction: len 1
            (1, 1, cmd_mode(3)*modal_mode(6)=18, ego_fut_mode=6, 2)
        status: len 1
            (1, 1, 10)
        anchor_queue: len 4
            (1, 1, 11)
        period: ( 1, 11)
        """
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]  # [1, 900, 256]
        anchor_embed = det_output["anchor_embed"]  # [1, 900, 256]
        """
        len(det_output['classification']): 6
        det_output['classification'][-1].shape: torch.Size([1, 900, 10])
        """
        det_classification = det_output["classification"][-1].sigmoid()
        """
        len(det_output['prediction']): 6
        det_anchors = det_output['prediction'][-1].shape: torch.Size([1, 900, 11])
        """
        det_anchors = det_output["prediction"][-1]  # (1, 900, 11)
        det_confidence = det_classification.max(dim=-1).values  # (1, 900)
        """
        self.num_det: 50
        instance_feature_selected: [1, 50, 256]
        anchor_embed_selected: [1, 50, 256]
        """
        _, (instance_feature_selected,
            anchor_embed_selected) = topk(det_confidence, self.num_det,
                                          instance_feature, anchor_embed)
        ############# map ##################
        map_instance_feature = map_output["instance_feature"]  # [1, 100, 256]
        map_anchor_embed = map_output["anchor_embed"]  # [1, 100, 256]
        map_classification = map_output["classification"][-1].sigmoid()
        """
        len(map_output['prediction']): 6
        map_anchors = map_output['prediction'][-1].shape: torch.Size([1, 100, 40])
        """
        map_anchors = map_output["prediction"][-1]
        """
        len(map_output['classification']): 6
        map_confidence = map_output['classification'][-1].shape: torch.Size([1, 100, 3])
            차선, 경계, 보행자 횡단보도
        """
        map_confidence = map_classification.max(dim=-1).values
        """
        self.num_map: 10
        map_instance_feature_selected: [1, 10, 256]
        map_anchor_embed_selected: [1, 10, 256]
        """
        _, (map_instance_feature_selected,
            map_anchor_embed_selected) = topk(map_confidence, self.num_map,
                                              map_instance_feature,
                                              map_anchor_embed)

        # =========== get ego/temporal feature/anchor ===========
        # import ipdb;ipdb.set_trace()
        bs, num_anchor, dim = instance_feature.shape
        device = instance_feature.device
        (
            ego_feature,  # [1, 1, 256]
            ego_anchor,  # [1, 1, 11] # [X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]
            temp_instance_feature,  # [1, 901, 4, 256] # 자차와 주변 장애물들의 feature을 max_queue_length 만큼 저장한 것
            temp_anchor,  # [1, 901, 4, 11] # 자차와 주변 장애물들의 앵커 정보를 max_queue_length 만큼 저장한 것
            temp_mask,  # [1, 901, 4] # 자차와 주변 장애물들의 mask 정보를 max_queue_length 만큼 저장한 것
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,  # InstanceBank.anchor_handler(=SparsePoint3DKeyPointsGenerator) of det_head
        )
        # ego_anchor_embed: [1, 1, 256] # ego_anchor: [1, 1, 11]
        # anchor_encoder = SparseBox3DEncoder
        ego_anchor_embed = anchor_encoder(ego_anchor)  # (1, 1, 256)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(
            0, 1)  # (b, 901, 4, 256) -> (b*901, 4, 256)
        temp_anchor_embed = temp_anchor_embed.flatten(
            0, 1)  # (b, 901, 4, 256) -> (b*901, 4, 256)
        temp_mask = temp_mask.flatten(0, 1)  # (b, 901, 4) -> (b*901, 4)
        # import ipdb;ipdb.set_trace()
        # =========== mode anchor init ===========
        """
        det_classification
            det_output['classification'][-1].shape: torch.Size([1, 900, 10])
        det_anchors
            det_output['prediction'][-1].shape: torch.Size([1, 900, 11])
        """
        """motion_anchor: [1, 900, 6, 12, 2] ( 세계 좌표계 , 아니면 자차기준 좌표계)
        주변 900개 객체들의 움직임을 예측하기 위한 초기 anchor
            한 객체마다 (6, 12, 2) 의 초기 anchor
        """
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        # (6, 6, 2) -> (1, 6, 6, 2) -> (bs, 1, 6, 6, 2) # bs, cmd_mode, modal_mode, ego_fut_ts, 2
        """ # bs, cmd_mode, modal_mode, ego_fut_ts, 2
            (3, 6, 6, 2) -> (1, 3, 6, 6, 2) -> (1, 3, 6, 6, 2)
        """
        plan_anchor = torch.tile(self.plan_anchor[None], (bs, 1, 1, 1, 1))
        # =========== mode query init ===========
        """ motion_anchor: [1, 900, 6, 12, 2]
        motion_anchor[..., -1, :]: [1, 900, 6, 2] (마지막 점의 위치를 말하는 것!)
        gen_sineembed_for_position(motion_anchor[..., -1, :]).shape: torch.Size([1, 900, 6, 256])
        motion_mode_query: [1, 900, 6, 256]
        """
        motion_mode_query = self.motion_anchor_encoder(
            gen_sineembed_for_position(motion_anchor[..., -1, :]))
        # print("plan_anchor.shape", plan_anchor.shape) # [1, 1, 6, 6, 2]
        """ # bs, cmd_mode, modal_mode, ego_fut_ts, 2
            (1, 3, 6, 6, 2) -> (1, 3, 6, 2) -(gen_sineembed_for_position)-> (1, 3, 6, 256)
            a = plan_anchor_encoder -> (1, 3, 6, 256)
            plan_mode_query -> (1, 1, 18, 256) # bs, 1, cmd_mode*modal_mode, 256
        """
        plan_pos = gen_sineembed_for_position(
            plan_anchor[..., -1, :])  # (1, 3, 6, 256)
        a = self.plan_anchor_encoder(plan_pos)  # (1, 1, 6, 256)
        plan_mode_query = a.flatten(1, 2).unsqueeze(1)

        # # =========== plan query init ===========
        # gt_ego_fut_trajs = metas['gt_ego_fut_trajs']
        # odo_info_fut = self.normalize_ego_fut_trajs(gt_ego_fut_trajs)
        # timesteps = torch.randint(
        #     0, self.diffusion_scheduler.config.num_train_timesteps,
        #     (bs,), device=device
        # )# TODO, only bs timesteps
        # noise = torch.randn(odo_info_fut.shape, device=device)
        # noisy_traj_points = self.diffusion_scheduler.add_noise(
        #     original_samples=odo_info_fut,
        #     noise=noise,
        #     timesteps=timesteps,
        # ).float()
        # noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        # noisy_traj_points = self.denormalize_ego_fut_trajs(noisy_traj_points) # bs, ego_fut_ts, 2
        # noisy_traj_points_cum = noisy_traj_points.cumsum(dim=-2)
        # traj_feature = self.pool_feature_from_traj(noisy_traj_points_cum,metas,feature_maps) # bs,1,2
        # import ipdb;ipdb.set_trace()
        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat(
            [instance_feature_selected, ego_feature], dim=1)  # [B, 51, 256]
        anchor_embed_selected = torch.cat(
            [anchor_embed_selected, ego_anchor_embed], dim=1)  # [B, 51, 256]

        instance_feature = torch.cat([instance_feature, ego_feature],
                                     dim=1)  # (1, 901, 256)
        # print("anchor_embed.shape", anchor_embed.shape) # [1, 900, 256]
        # print("ego_anchor_embed.shape", ego_anchor_embed.shape) # [1, 1, 256]
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed],
                                 dim=1)  # [1, 901, 256]
        # print("anchor_embed.shape", anchor_embed.shape) # [1, 901, 256]
        # TODO, need to add traj anchor embed into temp_anchor_embed & temp_instance_feature

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        # planning_diffusion_loss = []
        # import ipdb;ipdb.set_trace()
        for i, op in enumerate(self.interact_operation_order):
            if self.interact_layers[i] is None:
                continue
            elif op == "temp_gnn":
                """
                각 객체가, 자신의 과거 feature과 self attention
                """
                # input: (b, 901, 256)
                # output:(b*901, 1, 256)
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0,
                                             1).unsqueeze(1),  # (b*901, 1, 256)
                    temp_instance_feature,  # (b*901, 4, 256)
                    temp_instance_feature,  # (b*901, 4, 256)
                    query_pos=anchor_embed.flatten(
                        0, 1).unsqueeze(1),  # (b*901, 1, 256)
                    key_pos=temp_anchor_embed,  # (b*901, 4, 256)
                    key_padding_mask=temp_mask,  # (b*901, 4)
                )
                # instance_feature : (b, 901, 256)
                instance_feature = instance_feature.reshape(
                    bs, num_anchor + 1, dim)
            elif op == "gnn":
                """
                각 객체가, 높은 확률의 객체들과 cross attention
                """
                instance_feature = self.graph_model(
                    i,
                    instance_feature,  # (b, 901, 256)
                    instance_feature_selected,  # [B, 51, 256]
                    instance_feature_selected,  # [B, 51, 256]
                    query_pos=anchor_embed,  # # [B, 901, 256]
                    key_pos=anchor_embed_selected,  # [B, 51, 256]
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.interact_layers[i](instance_feature)
            elif op == "cross_gnn":
                """
                각 객체가, 높은 확률의 map 객체들과 cross attention
                """
                instance_feature = self.interact_layers[i](
                    instance_feature,  # (b, 901, 256)
                    key=map_instance_feature_selected,  # [B, 10, 256]
                    query_pos=anchor_embed,  # [B, 901, 256]
                    key_pos=map_anchor_embed_selected,  # [B, 10, 256]
                )
            elif op == "refine":
                """
                motion_mode_query: (1, 900, 6, 256)
                instance_feature: (1, 901, 256)
                anchor_embed: (1, 901, 256)
                
                motion_query.shape: (1, 900, 6, 256) + (1, 900, 1, 256) -> (1, 900, 6, 256)
                
                """
                """
                    plan_mode_query -> (1, 1, 18, 256) # bs, 1, cmd_mode(=3)*modal_mode, 256
                    plan_query.shape: (1, 1, 18, 256) + (1, 1, 1, 256) -> (1, 1, 18, 256)
                """
                motion_query = motion_mode_query + (
                    instance_feature +
                    anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed
                                               )[:, num_anchor:].unsqueeze(2)
                (
                    motion_cls,  # [B, 900, 6]
                    motion_reg,  # [B, 900, 6, 12, 2]
                    plan_cls,  # None
                    plan_reg,  # None
                    plan_status,  # [B, 1, 10]
                ) = self.interact_layers[i](
                    motion_query,  # [1, 900, 6, 256]
                    plan_query,  # [1, 1, 18, 256]
                    instance_feature[:, num_anchor:],  # [1, 1, 256]
                    anchor_embed[:, num_anchor:],  # [1, 1, 256]
                    metas,
                )
                motion_classification.append(motion_cls)  # [B, 900, 6]
                motion_prediction.append(
                    motion_reg
                )  # [B, 900, fut_mode, fut_ts, 2] (예: [B, 900, 6, 12, 2])
                planning_classification.append(plan_cls)  # None
                planning_prediction.append(plan_reg)  # None
                planning_status.append(plan_status)  # [B, 1, 10]
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor],
                                         det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:],
                                           plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        # =================== diffusion part ====================
        step_num = 2
        # global_cond=(instance_feature + anchor_embed)[:, :num_anchor].mean(dim=1)
        # _, (instance_feature_selected, anchor_embed_selected) = topk(
        #     det_confidence, self.num_det, instance_feature[:,:num_anchor], anchor_embed[:,:num_anchor]
        # )
        device = instance_feature.device
        # img = torch.randn(bs*self.ego_fut_mode, self.ego_fut_ts, 2).to(device)
        # import ipdb;ipdb.set_trace()
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 40 / step_num  # 40 / 2 = 20
        a = np.arange(0, step_num)  # a = [0, 1]
        b = (a * step_ratio).round()  # b = [0, 20]
        b_reversed = b[::-1]  # b_reversed = [20, 0]
        roll_timesteps = b_reversed.copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        # truncate the timesteps to 40
        # print("plan_query.shape", plan_query.shape) # [1, 1, 6, 256]
        """
        weight
            plan_query.shape: (1, 1, 18, 256) + (1, 1, 1, 256) -> (1, 1, 18, 256)  # bs, 1, cmd_mode(=3)*modal_mode, 256
            plan_nav_query -> (1, 18, 256) # (bs, cmd_mode*modal_mode, 256)
            plan_nav_query -> (1, 3, 6(=self.ego_fut_mode), 256) 
        """
        plan_nav_query = plan_query.squeeze(1)  # (1, 18, 256)
        # ego_fut_mode: 6
        plan_nav_query = plan_nav_query.view(bs, 3, self.ego_fut_mode,
                                             -1)  # (1, 3, 6, 256)
        bs_indices = torch.arange(bs, device=plan_query.device)  # [0]
        """
        print("metas['gt_ego_fut_cmd'].shape:", metas['gt_ego_fut_cmd'].shape) # [1, 3]
        print("metas['gt_ego_fut_cmd'].argmax(dim=-1).shape:", metas['gt_ego_fut_cmd'].argmax(dim=-1).shape) # [1]
        """
        cmd = metas['gt_ego_fut_cmd'].argmax(dim=-1)
        """
        cmd_plan_nav_query: ego future command gt를 이용하여 만든 선별된 query
        """
        cmd_plan_nav_query = plan_nav_query[bs_indices,
                                            cmd]  # (b, self.ego_fut_mode, 256)
        # import ipdb;ipdb.set_trace()
        # plan_anchor: (1, 3, 6, 6, 2)
        cmd_plan_anchor = plan_anchor[bs_indices, cmd]  # (b, 6, 6, 2)
        zeros_cat = torch.zeros(bs, 6, 1, 2, device=device)  # (b, 6, 1, 2)
        cmd_plan_anchor = torch.cat([zeros_cat, cmd_plan_anchor],
                                    dim=2)  # (b, 6, 7, 2)
        tgt_cmd_plan_anchor = cmd_plan_anchor[:, :,
                                              1:, :] - cmd_plan_anchor[:, :, :
                                                                       -1, :]  # (b, 6, 6, 2)

        diff_planning_prediction = []
        diff_planning_classification = []
        # ego_anchor_embed: (1, 1, 256)
        repeat_ego_anchor_embed = ego_anchor_embed.repeat(
            1, self.ego_fut_mode, 1)  # (1, 6, 256)

        img = self.normalize_ego_fut_trajs(tgt_cmd_plan_anchor)  # (b, 6, 6, 2)
        noise = torch.randn(img.shape, device=device)  # (b, 6, 6, 2)
        trunc_timesteps = torch.ones(
            (bs,), device=device,
            dtype=torch.long) * 8  # [8, ..., 8] # 길이 = bs -> [8]
        img = self.diffusion_scheduler.add_noise(original_samples=img,
                                                 noise=noise,
                                                 timesteps=trunc_timesteps)
        img = img.view(bs * self.ego_fut_mode, self.ego_fut_ts,
                       2)  # (b*6, 6, 2)

        # import ipdb;ipdb.set_trace()
        # roll_timesteps: [20, 0]
        # roll_timesteps[:] = [20, 0]
        for k in roll_timesteps[:]:
            # k = 20, 0
            x_boxes = torch.clamp(img, min=-1, max=1)  # (b*6, 6, 2)
            noisy_traj_points = self.denormalize_ego_fut_trajs(
                x_boxes)  # (b*6, 6, 2)
            # TODO: 여기서부터
            traj_pos_embed = gen_sineembed_for_position(
                noisy_traj_points, hidden_dim=128)  # (b*6, 6, 128)
            traj_pos_embed = traj_pos_embed.flatten(-2)  # (b*6, 6*128)
            # traj_feature: (b*6, 256)
            traj_feature = self.plan_pos_encoder(traj_pos_embed)  # (b*6, 768)
            # traj_feature: (b*6, 256) -> (b, 6, 256)
            traj_feature = traj_feature.view(bs, self.ego_fut_mode,
                                             -1)
            # noisy_traj_points_cum = noisy_traj_points.cumsum(dim=-2)
            # traj_feature = self.pool_feature_from_traj(noisy_traj_points_cum,metas,feature_maps)
            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps],
                                         dtype=torch.long,
                                         device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
                """
                timesteps.shape: 1
                timesteps: tensor([20], device='cuda:0')
                """
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(img.shape[0])  # img.shape[0]: b*6
            """
            timesteps.shape: b*6
            timesteps: [20, 20, 20, 20, 20, 20]
            time_embed: [b*6, 256]
            """
            time_embed = self.time_mlp(timesteps)
            # import ipdb;ipdb.set_trace()
            diff_plan_reg = noisy_traj_points  # (b*6, 6, 2)
            for i, op in enumerate(self.diff_operation_order):
                if self.diff_layers[i] is None:
                    continue
                elif op == "traj_pooler":
                    # import ipdb;ipdb.set_trace()
                    if len(diff_plan_reg.shape) != 3:
                        """
                        diff_plan_reg: (b, 1, 3*6, 6, 2)
                        """
                        a = diff_plan_reg[
                            :,
                            :,
                            -self.ego_fut_mode:,
                        ] # (b, 1, 6, 6, 2)
                        # import ipdb;ipdb.set_trace()
                        diff_plan_reg = a.flatten(0, 2) # (b*6, 6, 2)
                    """
                    traj_feature: (b, 6, 256)
                    diff_plan_reg: (b*6, 6, 2)
                    feature_maps : List[Tensor]
                            [0]: (1, 89760, 256)
    – 여러 스케일(또는 레벨)과 카메라에서 추출된 feature map들을 공간적 위치(픽셀 또는 패치) 단위로 평탄화(flatten)하여 하나의 큰 텐서로 연결한 결과
    – 여기서 89760는 모든 카메라와 모든 스케일의 픽셀(또는 패치) 수의 총합이고, 256는 각 위치에서의 feature 채널 수를 의미
    – 즉, 이 텐서는 배치와 카메라 차원을 합쳐서 모든 spatial 위치의 특징들을 한 번에 처리할 수 있도록 만들어진 “열(feature column)” 형태의 표현        
                            [1]: (6, 4, 2)
    – 각 카메라(6)와 각 스케일(레벨, 4)에서 원래의 공간적 크기(높이, 너비)를 나타내는 정보
    – 여기서 6는 카메라의 수, 4는 각 카메라에서 사용한 스케일(또는 레벨)의 수, 2는 각각 (높이, 너비)를 의미
    – 이 정보는 평탄화된 col_feats를 다시 원래의 spatial 구조로 복원할 때 기준으로 사용
                            [2]: (6, 4)
    – 각 카메라별, 각 스케일별로 평탄화된 col_feats 내에서 해당 스케일의 feature들이 시작하는 인덱스를 나타냅
    – (6, 4)에서 6는 카메라 수, 4는 각 카메라에서의 스케일 수를 의미하며, 
        이 값들을 이용해 분할된 feature map 들을 다시 각 스케일 단위로 분리하거나 재구성할 수 있음
                    """
                    # V1TrajPooler : 카메라 feature map과 traj_feature간 cross attention (Deformable Aggregation)
                    # traj_feature: (b, 6, 256)
                    traj_feature = self.diff_layers[i](
                        traj_feature,
                        diff_plan_reg, # (b*6, 6, 2)
                        metas,
                        feature_maps,
                        modal_num=self.ego_fut_mode)
                    # import ipdb;ipdb.set_trace()
                elif op == "self_attn":
                    traj_feature = self.diff_layers[i](
                        traj_feature,
                        traj_feature,
                        traj_feature,
                    )
                elif op == "modulation":
                    """ 이를 통해 모델은 시간에 따른 변화에 traj_feature 가 유연하게 대응할 수 있게 됩니다."""
                    # V1ModulationLayer
                    # traj_feature: (b, 6, 256)
                    traj_feature = self.diff_layers[i](
                        traj_feature, # (b, 6, 256)
                        time_embed.view(bs, self.ego_fut_mode, -1), # [b*6, 256] -> [b, 6, 256]
                        global_cond=global_cond
                        if self.diff_layers[i].if_global_cond else None,
                    )
                elif op == "agent_cross_gnn":
                    """각 궤적(modal)이, 주변 차량(자차 포함) 과 cross attention"""
                    # traj_feature: (b, 6, 256)
                    traj_feature = self.diff_graph_model(
                        i,
                        traj_feature,  # (b, 6, 256)
                        instance_feature_selected,  # (b, 51, 256)
                        instance_feature_selected,  # (b, 51, 256)
                        query_pos=repeat_ego_anchor_embed,  # (1, 6, 256)
                        key_pos=anchor_embed_selected,  # # [B, 51, 256]
                    )
                elif op == "map_cross_gnn":
                    traj_feature = self.diff_layers[i](
                        traj_feature,
                        map_instance_feature_selected,
                        map_instance_feature_selected,
                        query_pos=repeat_ego_anchor_embed,
                        key_pos=map_anchor_embed_selected,
                    )
                elif op == "anchor_cross_gnn":
                    """cmd_plan_nav_query
                    (clustering으로 추출한) plan_anchor (b, 3, 6, 256)에, 아래 2가지를 더한 값에,
                        ego_instance_feature (b, 1, 1, 256)
                        ego_anchor_embed (b, 1, 1, 256)
                            참고로 이 두 개는 과거 "자차, 주변 차량들" + "map 정보"까지 반영된 정보임
                    meta['gt_ego_fut_cmd']를 반영하여, (b, 6, 256) shape을 띠는 벡터
                    
                    """
                    # traj_feature: (b, 6, 256)
                    traj_feature = self.diff_layers[i](
                        traj_feature,  # (b, 6, 256)
                        key=cmd_plan_nav_query,  # (b, self.ego_fut_mode, 256)
                        value=cmd_plan_nav_query,  # (b, self.ego_fut_mode, 256)
                    )
                elif op == "norm" or op == "ffn":
                    traj_feature = self.diff_layers[i](traj_feature)
                elif op == "diff_refine":
                    # import ipdb;ipdb.set_trace()
                    """
            diff_plan_reg: (b, 1, 3*6, 6, 2)
            diff_plan_cls: (b, 1, 3*6)
                    """
                    diff_plan_reg, diff_plan_cls = self.diff_layers[i](
                        traj_feature,) # (b, 6, 256)
            # import ipdb;ipdb.set_trace()
            if len(diff_plan_reg.shape) != 3:
                # import ipdb;ipdb.set_trace()
                a = diff_plan_reg[
                    :,
                    :,
                    -self.ego_fut_mode:,
                ] # (b, 1, 6, 6, 2)
                diff_plan_reg = a.flatten(0, 2) # (b*6, 6, 2)
            x_start = diff_plan_reg #(b*6, 6, 2)
            x_start = self.normalize_ego_fut_trajs(x_start) # (b*6, 6, 2)
            # inverse diffusion step (remove noise)
            """
            diffusion network가 x_0를 출력 = x_start
            
            .prev_sample -> x_t-1 을 구하겠다는 뜻
            """
            img = self.diffusion_scheduler.step(model_output=x_start, # (b*6, 6, 2) # x_0
                                                timestep=k, # k = 20, 0
                                                sample=img # (b*6, 6, 2) # x_t
                                                ).prev_sample
        # diffusion_output_x = img[:, :, 0:1]
        # diffusion_output_x = diffusion_output_x * 3

        # diffusion_output_y = img[:, :, 1:2]
        # diffusion_output_y = (diffusion_output_y + 1) / 2 * 8.1 - 0.5
        # plan_reg = torch.cat([diffusion_output_x, diffusion_output_y], dim=2)
        # diff_plan_reg: (b*6, 6, 2)
        # a: (b, 1, 6, 6, 2)
        a = diff_plan_reg.reshape(bs, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        #
        b = a.repeat(1,3,1,1,1)
        c = b.view(bs,1,3*self.ego_fut_mode,self.ego_fut_ts,2)
        diff_planning_prediction.append(c)
        diff_planning_classification.append(diff_plan_cls)
        planning_output["prediction"] = diff_planning_prediction
        planning_output["classification"] = diff_planning_classification

        return motion_output, planning_output

    def loss(self, motion_model_outs, planning_model_outs, data,
             motion_loss_cache):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data,
                                       motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            (cls_target, cls_weight, reg_pred, reg_target, reg_weight,
             num_pos) = self.motion_sampler.sample(
                 reg,
                 data["gt_agent_fut_trajs"],
                 data["gt_agent_fut_masks"],
                 motion_loss_cache,
             )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls,
                                            cls_target,
                                            weight=cls_weight,
                                            avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(reg_pred,
                                            reg_target,
                                            weight=reg_weight,
                                            avg_factor=num_pos)

            output.update({
                f"motion_loss_cls_{decoder_idx}": cls_loss,
                f"motion_loss_reg_{decoder_idx}": reg_loss,
            })

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        # import ipdb;ipdb.set_trace()
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        # diffusion_losses = model_outs["diffusion_loss"]
        diffusion_predictions = model_outs["diffusion_prediction"]
        diffusion_classification = model_outs["diffusion_classification"]
        tgt_cmd_plan_anchor = model_outs["tgt_cmd_plan_anchor"]
        output = {}
        # import ipdb;ipdb.set_trace()
        for decoder_idx, (cls, reg, status) in enumerate(
                zip(cls_scores, reg_preds, status_preds)):
            if cls is None and reg is None:
                status_loss = self.plan_loss_status(status.squeeze(1),
                                                    data['ego_status'])
                output.update({
                    # f"planning_loss_cls_{decoder_idx}": cls_loss,
                    # f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                    # f"planning_loss_reg_diffusion_{decoder_idx}": diff_reg_loss,
                })
            else:
                (
                    cls,
                    cls_target,
                    cls_weight,
                    reg_pred,
                    reg_target,
                    reg_weight,
                ) = self.planning_sampler.sample(
                    cls,
                    reg,
                    data['gt_ego_fut_trajs'],
                    data['gt_ego_fut_masks'],
                    data,
                )
                cls = cls.flatten(end_dim=1)
                cls_target = cls_target.flatten(end_dim=1)
                cls_weight = cls_weight.flatten(end_dim=1)
                cls_loss = self.plan_loss_cls(cls,
                                              cls_target,
                                              weight=cls_weight)

                reg_weight = reg_weight.flatten(end_dim=1)
                reg_pred = reg_pred.flatten(end_dim=1)
                reg_target = reg_target.flatten(end_dim=1)
                reg_weight = reg_weight.unsqueeze(-1)

                reg_loss = self.plan_loss_reg(reg_pred,
                                              reg_target,
                                              weight=reg_weight)
                status_loss = self.plan_loss_status(status.squeeze(1),
                                                    data['ego_status'])

                output.update({
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                    # f"planning_loss_reg_diffusion_{decoder_idx}": diff_reg_loss,
                })
        # import ipdb;ipdb.set_trace()
        for decoder_idx, (diffusion_prediction,
                          diffusion_classification) in enumerate(
                              zip(diffusion_predictions,
                                  diffusion_classification)):

            (
                cls,
                cls_target,
                cls_weight,
                reg_pred,
                reg_target,
                reg_weight,
            ) = self.planning_sampler.sample(
                diffusion_classification,
                diffusion_prediction,
                tgt_cmd_plan_anchor,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(reg_pred,
                                          reg_target,
                                          weight=reg_weight)

            output.update({
                f"planning_loss_reg_diffusion_{decoder_idx}": reg_loss,
                f"planning_loss_cls_diffusion_{decoder_idx}": cls_loss,
            })
        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        """ det_output : Dict
        len(det_output['classification']): 6
            det_output['classification'][-1].shape: torch.Size([1, 900, 10])
        len(det_output['prediction']): 6
            det_output['prediction'][-1].shape: torch.Size([1, 900, 11])
        len(det_output['quality']): 6
            det_output['quality'][-1].shape: torch.Size([1, 900, 2])
        """
        """ motion_output : Dict
        "classification": len = 1
            (1, 900, fut_mode=6)
        "prediction": len = 1
            (1, 900, fut_mode=6, fut_ts=12, 2)
        "period": (1, 900)
        "anchor_queue": len = 4
            (1, 900, 11)
        """
        """ planning_output : Dict
        classification: len 1
            (1, 1, cmd_mode(3)*modal_mode(6)=18)
        prediction: len 1
            (1, 1, cmd_mode(3)*modal_mode(6)=18, ego_fut_mode=6, 2)
        status: len 1
            (1, 1, 10)
        anchor_queue: len 4
            (1, 1, 11)
        period: ( 1, 11)
        """
        """ motion_result: List, len = 1 (아마 batch size 만큼 나올 것)
        dict
            trajs_3d: (300, fut_mode=6, fut_ts=12, 2)
            trajs_score: (300, fut_mode=6)
            anchor_queue: (300, max_length(=4), 10)
            period: (300)
        
        """
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),  # [1, 900]
            det_output.get("quality"),
            motion_output,
        )
        """ planning result: len = 1 (아마 batch size 만큼 나올 것)
        dict
            planning_score: (cmd_mode=3, modal_mode=6)
            planning: (cmd_mode=3, modal_mode=6, ego_fut_mode=6, 2)
            final_planning: (ego_fut_mode=6, 2)
            ego_period: (1)
            ego_anchor_queue: (1, max_length=4, 10)
        """
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output,
            data,
        )

        return motion_result, planning_result
