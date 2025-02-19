from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


@DETECTORS.register_module()
class V1SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
    ):
        super(V1SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)
        model_outs = self.head(feature_maps, data)
        """ model_outs: det_output, map_output, motion_output, planning_output
         motion_output : Dict
         "classification": len = 1
             (1, 900, fut_mode=6)
         "prediction": len = 1
             (1, 900, fut_mode=6, fut_ts=12, 2)
         "period": (1, 900)
         "anchor_queue": len = 4
             (1, 900, 11)
        planning_output : Dict
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
        """
        results : list (len = b)
        output : list (len = b)
            output[0] : dict
            dict
                img_bbox
                    boxes_3d : torch.Size([300, 10])
                    scores_3d : torch.Size([300])
                    labels_3d : torch.Size([300])
                    cls_scores : torch.Size([300])
                    instance_ids : torch.Size([300])
                    vectors : list (len = 100)
                        각 원소는 torch.Size([20, 2])
                    trajs_3d : torch.Size([300, 6, 12, 2]) (motion_result)
                    trajs_score : torch.Size([300, 6]) (motion_result)
                    anchor_queue : torch.Size([300, 1, 10]) (motion_result)
                    period : torch.Size([300]) (motion_result)
                    planning_score : torch.Size([3, 6]) (planning_result)
                    planning : torch.Size([3, 6, 6, 2]) (planning_result)
                    final_planning : torch.Size([6, 2]) (planning_result)
                    ego_period : torch.Size([1]) (planning_result)
                    ego_anchor_queue : torch.Size([1, 1, 10]) (planning_result)


        
        """
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
