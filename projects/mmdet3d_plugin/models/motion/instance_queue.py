import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.core.box3d import *


@PLUGIN_LAYERS.register_module()
class InstanceQueue(nn.Module):

    def __init__(
        self,
        embed_dims,
        queue_length=0,
        tracking_threshold=0,
        feature_map_scale=None,
    ):
        super(InstanceQueue, self).__init__()
        self.embed_dims = embed_dims
        self.queue_length = queue_length  # history + current = 4
        self.tracking_threshold = tracking_threshold  # 0.2

        kernel_size = tuple([int(x / 2) for x in feature_map_scale])  #( 4, 11)
        # input:(1, 256, 8, 22) # output:(1, 256, 1, 1)
        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(embed_dims,
                      embed_dims,
                      3,
                      stride=1,
                      padding=1,
                      bias=False),  # (1, 256, 8, 22)
            nn.BatchNorm2d(embed_dims),
            nn.Conv2d(embed_dims,
                      embed_dims,
                      3,
                      stride=2,
                      padding=1,
                      bias=False),  # (1, 256, 4, 11)
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size),  # (1 256, 1, 1)
        )
        self.ego_anchor = nn.Parameter(
            torch.tensor([
                [
                    0, 0.5, -1.84 + 1.56 / 2,
                    np.log(4.08),
                    np.log(1.73),
                    np.log(1.56), 1, 0, 0, 0, 0
                ],
            ],
                         dtype=torch.float32),
            requires_grad=False,
        )  # (1, 11) shape

        self.reset()

    def reset(self):
        self.metas = None
        self.prev_instance_id = None
        self.prev_confidence = None
        self.period = None
        self.instance_feature_queue = []
        self.anchor_queue = []
        self.prev_ego_status = None
        self.ego_period = None
        self.ego_feature_queue = []
        self.ego_anchor_queue = []

    def get(
        self,
        det_output,
        feature_maps,
        metas,
        batch_size,
        mask,
        anchor_handler,
    ):
        """
        input
            det_output
                instance_feature # [1, 900, 256]
                prediction [-1] # [1, 900, 11]
                instance_id # [1, 900]
                classification[-1] # [1, 900, 10]
            feature_maps: List[Tensor]
                [0]: (1, 89760, 256)
                [1]: (6, 4, 2)
                [2]: (6, 4)
        output
            ego_feature: (B, 1, embed_dims)
                전방 카메라를 8 * 22 개의 token(각 token=256차원) 으로 나눈 feature을 CNN으로 잘 조합하여, 하나의 최종 token 화 한 것
            ego_anchor: (B, 1, 11)
                자차의 앵커 정보 (생성자에서 정의한 ego_anchor 의 VY 값은 이전 프레임의 회전 정보(또는 방향 정보를 암시하는 SIN_YAW)를 반영)
                11 = [X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]
            temp_instance_feature: (B, 901, max_queue_length, embed_dims)
                자차와 주변 장애물들의 feature을 max_queue_length 만큼 저장한 것
            temp_anchor: (B, 901, max_queue_length, 11)
                자차와 주변 장애물들의 앵커 정보를 max_queue_length 만큼 저장한 것
            temp_mask: (B, 901, max_queue_length)
                자차와 주변 장애물들의 mask 정보를 max_queue_length 만큼 저장한 것
        """
        """
        참고
            self.instance_feature_queue : 최대 길이가 self.queue_length 인 리스트
                리스트에 저장된 각 요소의 shape: [B, N, embed_dims]
            self.anchor_queue : 최대 길이가 self.queue_length 인 리스트
                리스트에 저장된 각 요소의 shape: [B, N, 11]
            self.period : [B, N]
                각 객체에 대해 누적된 기간(프레임 수)을 나타냄
        설명
            이전 프레임(또는 저장된 temporal 정보)에서 얻은 앵커(anchor)들이 현재 프레임의 좌표계로 보정되도록 하는 역할
            self.anchor_queue 와 self.ego_anchor_queue 에 저장된 앵커들을 현재 좌표계로 변환
        """

        if (self.period is not None and batch_size == self.period.shape[0]):
            if anchor_handler is not None:
                # 행렬(T_temp2cur)은 이전 좌표계에서 현재 좌표계로 변환하는 역할 shape: [B, 4, 4]
                T_temp2cur = feature_maps[0].new_tensor(
                    np.stack([
                        x["T_global_inv"]
                        @ self.metas["img_metas"][i]["T_global"]
                        for i, x in enumerate(metas["img_metas"])
                    ]))
                for i in range(len(self.anchor_queue)):
                    temp_anchor = self.anchor_queue[i]
                    temp_anchor = anchor_handler.anchor_projection(
                        temp_anchor,
                        [T_temp2cur],
                    )[0]
                    self.anchor_queue[i] = temp_anchor
                for i in range(len(self.ego_anchor_queue)):
                    temp_anchor = self.ego_anchor_queue[i]
                    temp_anchor = anchor_handler.anchor_projection(
                        temp_anchor,
                        [T_temp2cur],
                    )[0]
                    self.ego_anchor_queue[i] = temp_anchor
        else:
            self.reset()
        """
        아래의 것들을 업데이트
            self.instance_feature_queue : 최대 길이가 self.queue_length인 리스트
                리스트에 저장된 각 요소의 shape: [B, N, embed_dims]
            self.anchor_queue : 최대 길이가 self.queue_length인 리스트
                리스트에 저장된 각 요소의 shape: [B, N, 11]
            self.period : [B, N]
                각 객체에 대해 누적된 기간(프레임 수)을 나타냄
        """
        self.prepare_motion(det_output, mask)
        """
        ego_feature: (B, 1, embed_dims)
            전방 카메라를 8 * 22 개의 token(각 token=256차원) 으로 나눈 feature을 CNN으로 잘 조합하여, 하나의 최종 token 화 한 것
        ego_anchor: (B, 1, 11)
            자차의 앵커 정보 (생성자에서 정의한 ego_anchor의 VY 값은 이전 프레임의 회전 정보(또는 방향 정보를 암시하는 SIN_YAW)를 반영)
            11 = [X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]
        """
        ego_feature, ego_anchor = self.prepare_planning(feature_maps, mask,
                                                        batch_size)

        # temporal
        temp_instance_feature = torch.stack(self.instance_feature_queue, dim=2)
        temp_anchor = torch.stack(self.anchor_queue, dim=2)
        temp_ego_feature = torch.stack(self.ego_feature_queue, dim=2)
        temp_ego_anchor = torch.stack(self.ego_anchor_queue, dim=2)

        period = torch.cat([self.period, self.ego_period], dim=1)
        temp_instance_feature = torch.cat(
            [temp_instance_feature, temp_ego_feature], dim=1)
        temp_anchor = torch.cat([temp_anchor, temp_ego_anchor], dim=1)
        num_agent = temp_anchor.shape[1]

        temp_mask = torch.arange(len(self.anchor_queue),
                                 0,
                                 -1,
                                 device=temp_anchor.device)
        temp_mask = temp_mask[None, None].repeat((batch_size, num_agent, 1))
        temp_mask = torch.gt(temp_mask, period[..., None])

        return ego_feature, ego_anchor, temp_instance_feature, temp_anchor, temp_mask

    def prepare_motion(
        self,
        det_output,
        mask,
    ):
        """
        역할
            tracking_threshold 를 기준으로 tracking이 되는 앵커만 남김
            instance_feature_queue, anchor_queue, period에 정보를 저장
            총 길이: self.queue_length (4)
        아래의 것들을 업데이트
            self.instance_feature_queue : 최대 길이가 self.queue_length인 리스트
                리스트에 저장된 각 요소의 shape: [B, N, embed_dims]
            self.anchor_queue : 최대 길이가 self.queue_length인 리스트
                리스트에 저장된 각 요소의 shape: [B, N, 11]
            self.period : [B, N]
                각 객체에 대해 누적된 기간(프레임 수)을 나타냄
        """
        instance_feature = det_output["instance_feature"]  # [1, 900, 256]
        det_anchors = det_output["prediction"][-1]  # [1, 900, 11]

        if self.period == None:
            # self.period: [1, 900]
            self.period = instance_feature.new_zeros(
                instance_feature.shape[:2]).long()
        else:
            instance_id = det_output['instance_id']  #  # [1, 900]
            prev_instance_id = self.prev_instance_id  # [1, 900]
            match = instance_id[...,
                                None] == prev_instance_id[:,
                                                          None]  # [1, 900, 900]
            if self.tracking_threshold > 0:
                temp_mask = self.prev_confidence > self.tracking_threshold  # [1, 900]
                match = match * temp_mask.unsqueeze(1)  # [1, 900, 900]

            for i in range(len(self.instance_feature_queue)):
                temp_feature = self.instance_feature_queue[i]  # [1, 900, 256]
                temp_feature = (match[..., None] * temp_feature[:, None]).sum(
                    dim=2
                )  # [B, 900, 256] -> 과거 900개 앵커 중, tracking이 되는 앵커(instance_feature_queue)만 남김
                self.instance_feature_queue[i] = temp_feature

                temp_anchor = self.anchor_queue[i]
                temp_anchor = (match[..., None] * temp_anchor[:, None]).sum(
                    dim=2
                )  # [B, 900, 11] -> 과거 900개 앵커 중, tracking이 되는 앵커(anchor_queue)만 남김
                self.anchor_queue[i] = temp_anchor

            self.period = (match * self.period[:, None]).sum(
                dim=2)  # [B, 900] -> 과거 900개 앵커 중, tracking이 되는 앵커의 period만 남김

        self.instance_feature_queue.append(
            instance_feature.detach())  # [1, 900, 256]
        self.anchor_queue.append(det_anchors.detach())  # [1, 900, 11]
        self.period += 1

        if len(self.instance_feature_queue) > self.queue_length:
            self.instance_feature_queue.pop(0)
            self.anchor_queue.pop(0)
        self.period = torch.clip(self.period, 0, self.queue_length)

    def prepare_planning(
        self,
        feature_maps,
        mask,
        batch_size,
    ):
        """
        inputs
    `       feature_maps: List[Tensor]
                [0]: (1, 89760, 256)
                [1]: (6, 4, 2)
                [2]: (6, 4)
            mask: (B, ...)
        return
            ego_feature: (B, 1, embed_dims)
                전방 카메라를 8 * 22 개의 token(각 token=256차원) 으로 나눈 feature을 CNN으로 잘 조합하여, 하나의 최종 token 화 한 것
            ego_anchor: (B, 1, 11)
                자차의 앵커 정보 (생성자에서 정의한 ego_anchor의 VY 값은 이전 프레임의 회전 정보(또는 방향 정보를 암시하는 SIN_YAW)를 반영)
                11 = [X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]
        state update
            self.ego_feature_queue : 최대 길이가 self.queue_length인 리스트
                리스트에 저장된 각 요소의 shape: [B, 1, embed_dims]
            self.ego_anchor_queue : 최대 길이가 self.queue_length인 리스트
                리스트에 저장된 각 요소의 shape: [B, 1, 11]
        """
        ## ego instance init
        feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
        """
        feature_maps_inv: List[List[Tensor]], len(feature_maps_inv): 1
            feature_maps_inv[0]: List[Tensor], len(feature_maps_inv[0]): 4
                feature_maps_inv[0][0]: (1, 6, 256, 64, 176)
                feature_maps_inv[0][1]: (1, 6, 256, 32, 88)
                feature_maps_inv[0][2]: (1, 6, 256, 16, 44)
                feature_maps_inv[0][3]: (1, 6, 256, 8, 22)
        """
        feature_map = feature_maps_inv[0][-1][:, 0]  # (1, 256, 8, 22)
        # feature_map = input:(1, 256, 8, 22)
        # ego_feature = output:(1, 256, 1, 1)
        ego_feature = self.ego_feature_encoder(feature_map)
        ego_feature = ego_feature.unsqueeze(1).squeeze(-1).squeeze(
            -1)  # ( 1, 1, 256)

        ego_anchor = torch.tile(
            self.ego_anchor[None],
            (batch_size, 1, 1)  #  (B, 1, 11)
        )
        if self.prev_ego_status is not None:  # (B, 1, 11)
            prev_ego_status = torch.where(
                mask[:, None, None],
                self.prev_ego_status,
                self.prev_ego_status.new_tensor(0),
            )
            """자차 앵커의 y축 방향 속도 초기값으로, 이전 프레임의 회전 정보(또는 방향 정보를 암시하는 SIN_YAW)를 반영"""
            ego_anchor[..., VY] = prev_ego_status[..., 6]
            # ego_anchor: (B, 1, 11)

        if self.ego_period == None:
            self.ego_period = ego_feature.new_zeros(
                (batch_size, 1)).long()  # (B, 1)
        else:
            self.ego_period = torch.where(
                mask[:, None],
                self.ego_period,
                self.ego_period.new_tensor(0),
            )  # self.ego_period: (B, 1)

        self.ego_feature_queue.append(ego_feature.detach())
        self.ego_anchor_queue.append(ego_anchor.detach())
        self.ego_period += 1

        if len(self.ego_feature_queue) > self.queue_length:
            self.ego_feature_queue.pop(0)
            self.ego_anchor_queue.pop(0)
        self.ego_period = torch.clip(self.ego_period, 0, self.queue_length)

        return ego_feature, ego_anchor

    def cache_motion(self, instance_feature, det_output, metas):
        det_classification = det_output["classification"][-1].sigmoid(
        )  # [1, 900, 10]
        det_confidence = det_classification.max(dim=-1).values
        instance_id = det_output['instance_id']  # [1, 900]
        self.metas = metas
        self.prev_confidence = det_confidence.detach()
        self.prev_instance_id = instance_id

    def cache_planning(self, ego_feature, ego_status):
        """
        ego_feature: (B, 1, embed_dims)
        ego_status: (B, 1, 11)
        """
        self.prev_ego_status = ego_status.detach()
        self.ego_feature_queue[-1] = ego_feature.detach()
