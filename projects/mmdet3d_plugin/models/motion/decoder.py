from typing import Optional

import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.detection3d.decoder import *
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


@BBOX_CODERS.register_module()
class SparseBox3DMotionDecoder(SparseBox3DDecoder):

    def __init__(self):
        super(SparseBox3DMotionDecoder, self).__init__()

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        motion_output=None,
        output_idx=-1,
    ):
        """
        cls_scores: List (len=6) [ , ... ,  (b, 900, 10) ]
        box_preds: List (len=6) [ , ... ,  (b, 900, 11) ]
        instance_id: (b, 900)
        quality: List (len=6) [, ... (b, 1, 900, 2) ]
            - 2 : centerness and yawness indices in quality
        motion_output : Dict
            "classification": len = 1
                (1, 900, fut_mode=6)
            "prediction": len = 1
                (1, 900, fut_mode=6, fut_ts=12, 2)
            "period": (1, 900)
            "anchor_queue": len = 4
                (1, 900, 11)
        return
            output: List (len=b) [Dict ,... Dict]
                Dict
                    "trajs_3d": (num_output=300, fut_mode=6, fut_ts=12, 2)
                    "trajs_score": (num_output, fut_mode=6)
                    "anchor_queue": (num_output, 4, 10)
                    "period": (num_output)
        """
        squeeze_cls = instance_id is not None # True

        cls_scores = cls_scores[output_idx].sigmoid() # (b, 900, 10)

        if squeeze_cls:
            # cls_scores, cls_ids : (b, 900)
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            # cls_scores : (b, 900, 1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx] # (b, 900, 11)
        bs, num_pred, num_cls = cls_scores.shape
        a = cls_scores.flatten(start_dim=1) # a: (b, 900)
        # cls_scores: (b, num_output), indices: (b, num_output)
        cls_scores, indices = a.topk(self.num_output, dim=1, sorted=self.sorted)
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            """ quality (centerness)로 cls_scores, indices 보정
            0: centerness (객체 중심에 얼마나 가까운지, 즉 박스 품질의 신뢰도)
            1:  yawness (예측한 회전각의 신뢰도)
            """
            a = quality[output_idx] # (b, 900, 2)
            centerness = a[..., CNS] # CNS = 0 (b, 900)
            # centerness: (b, num_output)
            centerness = torch.gather(centerness, 1, indices // num_cls)
            # cls_scores_origin: (b, num_output)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid() # (b, num_output)
            # cls_scores, idx: (b, num_output)
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            # indices: (b, num_output)
            indices = torch.gather(indices, 1, idx)
        output = []
        anchor_queue = motion_output["anchor_queue"] #  len = 4, (b, 900, 11)
        anchor_queue = torch.stack(anchor_queue, dim=2) # (b, 900, 4, 11)
        period = motion_output["period"] # (b, 900)

        for i in range(bs):
            category_ids = cls_ids[i] # (900)
            if squeeze_cls:
                category_ids = category_ids[indices[i]] # (num_output)
            scores = cls_scores[i] # (num_output)
            box = box_preds[i, indices[i] // num_cls] # (num_output, 11)
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i] # (num_output)
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]
            # (num_output, 11) -> (num_output, 10)
            box = decode_box(box)
            trajs = motion_output["prediction"][-1] # (b, 900, fut_mode=6, fut_ts=12, 2)
            traj_cls = motion_output["classification"][-1].sigmoid() # (b, 900, fut_mode=6)
            traj = trajs[i, indices[i] // num_cls] # (num_output=300, fut_mode=6, fut_ts=12, 2)
            traj_cls = traj_cls[i, indices[i] // num_cls] # (num_output=300, fut_mode=6)
            if self.score_threshold is not None:
                traj = traj[mask[i]]
                traj_cls = traj_cls[mask[i]]
            # traj: (num_output=300, fut_mode=6, fut_ts=12, 2)
            traj = traj.cumsum(dim=-2) + box[:, None, None, :2] #
            """
            로직 (traj, traj_cls)
                - 예측한 900 객체의 미래 경로(trajs)와 "경로 class"(traj_cls) -> num_output=300개로 추림
                    - ("물체 class 확신 정도"가 높은 300개를 추림) + "물체 class 확신 정도" * "물체 중심의 신뢰도" 로 내림차순 정렬
                - box = decode_box(box)
                - traj가 변위 기준이었는데, 좌표 기준으로 바꾼 후, + (box 좌표 더해줘서) -> ego 차량 기준으로 변환
            """
            output.append({
                "trajs_3d": traj.cpu(), # (num_output=300, fut_mode=6, fut_ts=12, 2)
                "trajs_score": traj_cls.cpu() # (num_output, fut_mode=6)
            })

            temp_anchor = anchor_queue[i, indices[i] // num_cls] # (num_output=300, queue_len=4, 11)
            temp_period = period[i, indices[i] // num_cls] # (300)
            if self.score_threshold is not None:
                temp_anchor = temp_anchor[mask[i]]
                temp_period = temp_period[mask[i]]
            num_pred, queue_len = temp_anchor.shape[:2] # num_pred: 300, queue_len: 4
            temp_anchor = temp_anchor.flatten(0, 1) # (num_output * queue_len, 11)
            # temp_anchor: (num_output * queue_len, 10)
            temp_anchor = decode_box(temp_anchor)

            temp_anchor = temp_anchor.reshape(
                [num_pred, queue_len, box.shape[-1]]) # (num_output, queue_len, 10)
            """
            로직
                anchor_queue (900, 4, 11) -> temp_anchor (300, 4, 11)
                    - ("물체 class 확신 정도"가 높은 300개를 추림) + "물체 class 확신 정도" * "물체 중심의 신뢰도" 로 내림차순 정렬
                temp_anchor = decode_box(temp_anchor) -> (300, 4, 10)
            """
            output[-1]['anchor_queue'] = temp_anchor.cpu() # (num_output, 4, 10)
            output[-1]['period'] = temp_period.cpu() # (num_output)

        return output


@BBOX_CODERS.register_module()
class HierarchicalPlanningDecoder(object):

    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        use_rescore=False,
    ):
        super(HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_rescore = use_rescore

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        # import ipdb;ipdb.set_trace()
        classification = planning_output['classification'][
            -1]  # bs, 1, cmd_mode*modal_mode
        prediction = planning_output['prediction'][
            -1]  # bs,1,cmd_mode*modal_mode,ego_fut_ts,2
        bs = classification.shape[0]
        classification = classification.reshape(bs, 3, self.ego_fut_mode)
        prediction = prediction.reshape(bs, 3, self.ego_fut_mode,
                                        self.ego_fut_ts, 2).cumsum(dim=-2)
        # classification: bs, 3, modal_mode
        # final_planning: bs, ego_fut_ts, 2
        classification, final_planning = self.select(det_output, motion_output,
                                                     classification, prediction,
                                                     data)
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append({
                "planning_score": cls.sigmoid().cpu(),
                "planning": pred.cpu(),  # 3, modal_mode, ego_fut_ts, 2
                "final_planning": final_planning[i].cpu(),  # ego_fut_ts, 2
                "ego_period": period[i].cpu(),
                "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(),
            })

        return output

    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]

        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                det_confidence,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning

    def rescore(
        self,
        plan_cls,
        plan_reg,
        motion_cls,
        motion_reg,
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):

        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat

        def get_yaw(traj, start_yaw=np.pi / 2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)

        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode,
                                        self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[...,
                [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None,
                                                             None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None,
                                          None].repeat(1, 1, 1,
                                                       self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None,
                                                 [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode,
                                                1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1,
                                                    1).flatten(0, -2)

        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode,
                          ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False  # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls


@BBOX_CODERS.register_module()
class V1HierarchicalPlanningDecoder(object):

    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        use_rescore=False,
    ):
        super(V1HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_rescore = use_rescore

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        # import ipdb;ipdb.set_trace()
        # classification = planning_output['classification'][-1]
        prediction = planning_output['prediction'][-1]
        bs = prediction.shape[0]
        # classification = classification.reshape(bs, 3, self.ego_fut_mode)
        prediction = prediction.reshape(bs, self.ego_fut_ts, 2).cumsum(dim=-2)
        final_planning = prediction
        prediction = prediction[:, None,
                                None, :, :].repeat(1, 3, self.ego_fut_mode, 1,
                                                   1)
        classification = torch.zeros([bs, 3, self.ego_fut_mode])
        classification[:, :, -1] = 1
        # classification, final_planning = self.select(det_output, motion_output, classification, prediction, data)
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append({
                "planning_score": cls.sigmoid().cpu(),
                "planning": pred.cpu(),
                "final_planning": final_planning[i].cpu(),
                "ego_period": period[i].cpu(),
                "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(),
            })

        return output

    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]

        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                det_confidence,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning

    @staticmethod
    def rescore(
        # self,
        plan_cls,
        plan_reg,
        motion_cls,
        motion_reg,
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
        ego_fut_mode=6,
        ego_fut_ts=6,
    ):

        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat

        def get_yaw(traj, start_yaw=np.pi / 2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)

        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, ego_fut_mode, ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[...,
                [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None,
                                                             None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None,
                                          None].repeat(1, 1, 1, ego_fut_ts + 1,
                                                       2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None,
                                                 [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode,
                                                1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1,
                                                    1).flatten(0, -2)

        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode,
                          ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False  # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        # import ipdb;ipdb.set_trace()
        plan_cls = plan_cls + score_offset
        return plan_cls


@BBOX_CODERS.register_module()
class V2HierarchicalPlanningDecoder(HierarchicalPlanningDecoder):

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        classification = planning_output['classification'][-1]  # bs, modal_mode
        prediction = planning_output['prediction'][
            -1]  # bs,modal_mode,ego_fut_ts,2
        bs = classification.shape[0]
        classification = classification.reshape(bs, 1,
                                                self.ego_fut_mode).repeat(
                                                    1, 3, 1)
        prediction = prediction.reshape(bs, 1, self.ego_fut_mode,
                                        self.ego_fut_ts,
                                        2).cumsum(dim=-2).repeat(1, 3, 1, 1, 1)
        # classification: bs, 3, modal_mode
        # final_planning: bs, ego_fut_ts, 2
        classification, final_planning = self.select(det_output, motion_output,
                                                     classification, prediction,
                                                     data)
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append({
                "planning_score": cls.sigmoid().cpu(),
                "planning": pred.cpu(),  # 3, modal_mode, ego_fut_ts, 2
                "final_planning": final_planning[i].cpu(),  # ego_fut_ts, 2
                "ego_period": period[i].cpu(),
                "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(),
            })

        return output

    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]

        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                det_confidence,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning


def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection: 
            check if any corner point of boxes1 is inside boxes2 and vice versa.
        
        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision


def corners_in_box(boxes1, boxes2):
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack([
        torch.stack([cos_yaw, sin_yaw]),
        torch.stack([-sin_yaw, cos_yaw]),
    ])
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0]
                          >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1]
                          >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision
