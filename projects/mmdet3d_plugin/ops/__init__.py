import torch

from .deformable_aggregation import DeformableAggregationFunction


def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def feature_maps_format(feature_maps, inverse=False):
    if inverse:
        """
        로직
            inverse=True 옵션은 flatten되고 concat된 feature들을 원래의 카메라 및 스케일별 공간 형태로 복원하기 위한 단계
        
        input
            col_feats: (1, 89760, 256): Column Features
    – 여러 스케일(또는 레벨)과 카메라에서 추출된 feature map들을 공간적 위치(픽셀 또는 패치) 단위로 평탄화(flatten)하여 하나의 큰 텐서로 연결한 결과
    – 여기서 89760는 모든 카메라와 모든 스케일의 픽셀(또는 패치) 수의 총합이고, 256는 각 위치에서의 feature 채널 수를 의미
    – 즉, 이 텐서는 배치와 카메라 차원을 합쳐서 모든 spatial 위치의 특징들을 한 번에 처리할 수 있도록 만들어진 “열(feature column)” 형태의 표현          
            spatial_shape: (6, 4, 2): Spatial Shape
    – 각 카메라(6)와 각 스케일(레벨, 4)에서 원래의 공간적 크기(높이, 너비)를 나타내는 정보
    – 여기서 6는 카메라의 수, 4는 각 카메라에서 사용한 스케일(또는 레벨)의 수, 2는 각각 (높이, 너비)를 의미
    – 이 정보는 평탄화된 col_feats를 다시 원래의 spatial 구조로 복원할 때 기준으로 사용
            scale_start_index: (6, 4): Scale 시작 인덱스 정보
    – 각 카메라별, 각 스케일별로 평탄화된 col_feats 내에서 해당 스케일의 feature들이 시작하는 인덱스를 나타냅
    – (6, 4)에서 6는 카메라 수, 4는 각 카메라에서의 스케일 수를 의미하며, 
        이 값들을 이용해 분할된 feature map 들을 다시 각 스케일 단위로 분리하거나 재구성할 수 있음
        """
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, num_levels = spatial_shape.shape[:2]

        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()

        idx = 0
        cam_split = [1]
        cam_split_size = [sum(split_size[0])]
        for i in range(num_cams - 1):
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    if isinstance(feature_maps[0], (list, tuple)):
        formated = [feature_maps_format(x) for x in feature_maps]
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    col_feats = []
    for i, feat in enumerate(feature_maps):
        spatial_shape.append(feat.shape[-2:])
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)

    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    return feature_maps
