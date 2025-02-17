import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS

__all__ = ["MotionTarget", "PlanningTarget"]


def get_cls_target(
    reg_preds,
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum,
                             dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    return mode_idx


def get_best_reg(
    reg_preds,
    reg_target,
    reg_weight,
):
    bs, num_pred, mode, ts, d = reg_preds.shape
    reg_preds_cum = reg_preds.cumsum(dim=-2)
    reg_target_cum = reg_target.cumsum(dim=-2)
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum,
                             dim=-1)
    dist = dist * reg_weight.unsqueeze(2)
    dist = dist.mean(dim=-1)
    mode_idx = torch.argmin(dist, dim=-1)
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
    return best_reg


@BBOX_SAMPLERS.register_module()
class MotionTarget():

    def __init__(self,):
        super(MotionTarget, self).__init__()

    def sample(
        self,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        motion_loss_cache,
    ):
        bs, num_anchor, mode, ts, d = reg_pred.shape
        reg_target = reg_pred.new_zeros((bs, num_anchor, ts, d))
        reg_weight = reg_pred.new_zeros((bs, num_anchor, ts))
        indices = motion_loss_cache['indices']
        num_pos = reg_pred.new_tensor([0])
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(gt_reg_target[i]) == 0:
                continue
            reg_target[i, pred_idx] = gt_reg_target[i][target_idx]
            reg_weight[i, pred_idx] = gt_reg_mask[i][target_idx]
            num_pos += len(pred_idx)

        cls_target = get_cls_target(reg_pred, reg_target, reg_weight)
        cls_weight = reg_weight.any(dim=-1)
        best_reg = get_best_reg(reg_pred, reg_target, reg_weight)

        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos


@BBOX_SAMPLERS.register_module()
class PlanningTarget():

    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
        diffusion_loss=None,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, 3, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, 3, 1, self.ego_fut_mode,
                                    self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)
        # import ipdb;ipdb.set_trace()
        if diffusion_loss is not None:
            diffusion_loss = diffusion_loss.reshape(bs, 3, 1, self.ego_fut_mode,
                                                    -1)
            diffusion_loss = diffusion_loss[bs_indices, cmd]
            mode_idx = cls_target[..., None,
                                  None].repeat(1, 1, 1, self.ego_fut_ts * 2)
            diffusion_loss = torch.gather(diffusion_loss, 2,
                                          mode_idx).squeeze(2)
            diffusion_loss = diffusion_loss.mean()
            return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask, diffusion_loss
        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


@BBOX_SAMPLERS.register_module()
class V1PlanningTarget():

    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
    ):
        super(V1PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

    @staticmethod
    def get_cls_target(
        reg_preds,
        reg_target,
        reg_weight,
    ):
        bs, num_pred, mode, ts, d = reg_preds.shape
        reg_preds_cum = reg_preds.cumsum(dim=-2)
        reg_target_cum = reg_target.cumsum(dim=-2)
        dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum,
                                 dim=-1)
        dist = dist * reg_weight.unsqueeze(2)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        # mode_idx = torch.zeros(bs,num_pred, dtype=torch.int64).to(reg_preds.device)
        return mode_idx

    @staticmethod
    def get_best_reg(
        reg_preds,
        cls_target,
        reg_target,
        reg_weight,
    ):
        bs, num_pred, mode, ts, d = reg_preds.shape
        # reg_preds_cum = reg_preds.cumsum(dim=-2)
        # reg_target_cum = reg_target.cumsum(dim=-2)
        # dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
        # dist = dist * reg_weight.unsqueeze(2)
        # dist = dist.mean(dim=-1)
        # mode_idx = torch.argmin(dist, dim=-1)
        # mode_idx = torch.zeros(bs,num_pred, dtype=torch.int64).to(reg_preds.device)
        # mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
        mode_idx = cls_target[..., None, None, None].repeat(1, 1, 1, ts, d)
        best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
        return best_reg

    def sample(
        self,
        cls_pred,
        reg_pred,
        tgt_cmd_plan_anchor,
        gt_reg_target,
        gt_reg_mask,
        data,
        diffusion_loss=None,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, 3, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, 3, 1, self.ego_fut_mode,
                                    self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        # import ipdb;ipdb.set_trace()
        cls_target = self.get_cls_target(
            tgt_cmd_plan_anchor.view(bs, 1, self.ego_fut_mode, self.ego_fut_ts,
                                     2), gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = self.get_best_reg(reg_pred, cls_target, gt_reg_target,
                                     gt_reg_mask)
        # import ipdb;ipdb.set_trace()
        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


@BBOX_SAMPLERS.register_module()
class V2PlanningTarget():

    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
    ):
        super(V2PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

    @staticmethod
    def get_cls_target(
        reg_preds,
        reg_target,
        reg_weight,
    ):
        bs, num_pred, mode, ts, d = reg_preds.shape
        # reg_preds_cum = reg_preds.cumsum(dim=-2)
        # reg_target_cum = reg_target.cumsum(dim=-2)
        # dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
        # dist = dist * reg_weight.unsqueeze(2)
        # dist = dist.mean(dim=-1)
        # mode_idx = torch.argmin(dist, dim=-1)
        mode_idx = torch.zeros(bs, num_pred,
                               dtype=torch.int64).to(reg_preds.device)
        return mode_idx

    @staticmethod
    def get_best_reg(
        reg_preds,
        reg_target,
        reg_weight,
    ):
        bs, num_pred, mode, ts, d = reg_preds.shape
        # reg_preds_cum = reg_preds.cumsum(dim=-2)
        # reg_target_cum = reg_target.cumsum(dim=-2)
        # dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
        # dist = dist * reg_weight.unsqueeze(2)
        # dist = dist.mean(dim=-1)
        # mode_idx = torch.argmin(dist, dim=-1)
        mode_idx = torch.zeros(bs, num_pred,
                               dtype=torch.int64).to(reg_preds.device)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
        best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
        return best_reg

    def sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
        diffusion_loss=None,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        # cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, 1, self.ego_fut_mode, self.ego_fut_ts,
                                    2)
        # cls_pred = cls_pred[bs_indices, cmd]
        # reg_pred = reg_pred[bs_indices, cmd]
        # import ipdb;ipdb.set_trace()
        cls_target = self.get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = self.get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)
        # import ipdb;ipdb.set_trace()
        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


@BBOX_SAMPLERS.register_module()
class V3PlanningTarget():

    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
    ):
        super(V3PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

    @staticmethod
    def get_cls_target(
        reg_preds,
        reg_target,
        reg_weight,
    ):
        bs, num_pred, mode, ts, d = reg_preds.shape
        reg_preds_cum = reg_preds.cumsum(dim=-2)
        reg_target_cum = reg_target.cumsum(dim=-2)
        dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum,
                                 dim=-1)
        dist = dist * reg_weight.unsqueeze(2)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        # mode_idx = torch.zeros(bs,num_pred, dtype=torch.int64).to(reg_preds.device)
        return mode_idx

    @staticmethod
    def get_best_reg(
        reg_preds,
        cls_target,
        reg_target,
        reg_weight,
    ):
        bs, num_pred, mode, ts, d = reg_preds.shape
        # reg_preds_cum = reg_preds.cumsum(dim=-2)
        # reg_target_cum = reg_target.cumsum(dim=-2)
        # dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)
        # dist = dist * reg_weight.unsqueeze(2)
        # dist = dist.mean(dim=-1)
        # mode_idx = torch.argmin(dist, dim=-1)
        # mode_idx = torch.zeros(bs,num_pred, dtype=torch.int64).to(reg_preds.device)
        mode_idx = cls_target[..., None, None, None].repeat(1, 1, 1, ts, d)
        best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)
        return best_reg

    def sample(
        self,
        cls_pred,
        reg_pred,
        tgt_cmd_plan_anchor,
        gt_reg_target,
        gt_reg_mask,
        data,
        diffusion_loss=None,
    ):
        # import ipdb;ipdb.set_trace()
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        # cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, 1, self.ego_fut_mode, self.ego_fut_ts,
                                    2)
        # cls_pred = cls_pred[bs_indices, cmd]
        # reg_pred = reg_pred[bs_indices, cmd]
        # import ipdb;ipdb.set_trace()
        cls_target = self.get_cls_target(
            tgt_cmd_plan_anchor.view(bs, 1, self.ego_fut_mode, self.ego_fut_ts,
                                     2), gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = self.get_best_reg(reg_pred, cls_target, gt_reg_target,
                                     gt_reg_mask)
        # import ipdb;ipdb.set_trace()
        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask
