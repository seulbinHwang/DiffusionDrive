from shapely.geometry import LineString, box, Polygon
from shapely import ops, strtree

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union

class NuscMapExtractor(object):
    """NuScenes map ground-truth extractor.

    NuScenes 지도 데이터를 다양한 카테고리(예: 차선 경계, 도로 분할선, 보행자 횡단보도, 도로 경계 등)로 추출하고,
    필요한 경우 여러 개의 작은 폴리곤들을 병합하거나 분리하는 전처리 과정을 수행

    Args:
        data_root (str): path to nuScenes dataset
        roi_size (tuple or list): bev range
    """
    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        self.roi_size = roi_size
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            # 특정 지역의 지도 데이터를 로드
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc)
            """
            지도에서 특정 레이어(예: 차선, 도로, 보행자 횡단보도 등)의 정보를 쉽게 추출할 수 있도록 돕는 유틸리티 클래스
            """
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])
        
        # local patch in nuScenes format
        self.local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2, 
                roi_size[0] / 2, roi_size[1] / 2)
    
    def _union_ped(self, ped_geoms: List[Polygon]) -> List[Polygon]:
        ''' merge close ped crossings.
        
        Args:
            ped_geoms (list): list of Polygon
        
        Returns:
            union_ped_geoms (Dict): merged ped crossings 
        '''

        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()

            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = strtree.STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            for o in tree.query(pgeom):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        results = []
        for p in final_pgeom:
            results.extend(split_collections(p))
        return results
        
    def get_map_geom(self, 
                     location: str, 
                     translation: Union[List, NDArray],
                     rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `location` and self pose, self may be lidar or ego.
        선분
            'lane_divider' : 차선 분할선
                - 점선 혹은 실선
            'road_divider' : 도로 분할선
                - 도로의 다른 구역(예: 도로와 인도, 도로와 중앙 분리대) 사이의 구분을 제공
        다각형
            'ped_crossing' : 보행자 횡단보도
            `lane` : 차량이 실제로 주행할 수 있는 개별 차선의 면적을 나타내는 폴리곤
                - 각 차선의 경계(보통 차선 분할선과 도로의 외곽선)에 의해 한정됩니다.
            `road_segment` : drivable area 같은 거 (도로 구간)
                - 하나의 연속적인 도로 구간을 나타내는 폴리곤
                - **차선(lane)**이 여러 개 포함되어 있을 수 있는, 도로 전체의 주행 가능 영역을 포괄하는 개념

        Args:
            location (str): city name
            translation (array): self2global translation, shape (3,)
            rotation (array): self2global quaternion, shape (4, )
            
        Returns:
            geometries (Dict): extracted geometries by category.
        '''

        # (center_x, center_y, len_y, len_x) in nuscenes format
        patch_box = (translation[0], translation[1], 
                self.roi_size[1], self.roi_size[0])
        rotation = Quaternion(rotation)
        yaw = quaternion_yaw(rotation) / np.pi * 180

        # get dividers
        """ map_explorer
        지도에서 특정 레이어(예: 차선, 도로, 보행자 횡단보도 등)의 정보를 쉽게 추출할 수 있도록 돕는 유틸리티 클래스
        """
        lane_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw, 'lane_divider')
        
        road_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw, 'road_divider')
        
        all_dividers = []
        for line in lane_dividers + road_dividers:
            all_dividers += split_collections(line)

        # get ped crossings (횡단보도)
        ped_crossings = []
        # ped: 다각형(Polygon) 정보를 추출.
        ped = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'ped_crossing')
        
        for p in ped:
            # 추출된 폴리곤들을 split_collections로 개별 Polygon으로 분리
            ped_crossings += split_collections(p)
        # some ped crossings are split into several small parts
        # we need to merge them
        # _union_ped: 가까운 폴리곤들을 병합
        ped_crossings = self._union_ped(ped_crossings)
        
        ped_crossing_lines = []
        for p in ped_crossings:
            # extract exteriors to get a closed polyline
            """ get_ped_crossing_contour 
                각 보행자 횡단보도 영역에 대해 외곽선(LineString, 폐곡선 형태)을 추출
                이때 self.local_patch를 사용하여 ROI 내의 부분만 고려
            """
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_crossing_lines.append(line)

        # get boundaries
        # we take the union of road segments and lanes as drivable areas
        # we don't take drivable area layer in nuScenes since its definition may be ambiguous
        road_segments = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'road_segment')
        lanes = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'lane')
        union_roads = ops.unary_union(road_segments)
        union_lanes = ops.unary_union(lanes)
        drivable_areas = ops.unary_union([union_roads, union_lanes])
        """ split_collections를 통해 개별 다각형(Polygon) 리스트로 분리"""
        drivable_areas = split_collections(drivable_areas)
        
        # boundaries are defined as the contour of drivable areas
        """ 주행 가능 영역의 외곽선(LineString)을 추출, 이를 경계(boundary)로 정의"""
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        return dict(
            divider=all_dividers, # List[LineString]
            ped_crossing=ped_crossing_lines, # List[LineString]
            boundary=boundaries, # List[LineString]
            drivable_area=drivable_areas, # List[Polygon],
        )

