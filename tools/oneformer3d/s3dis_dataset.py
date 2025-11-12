from mmdet3d.registry import DATASETS
from mmdet3d.datasets.s3dis_dataset import S3DISDataset
import numpy as np


@DATASETS.register_module()
class S3DISSegDataset_(S3DISDataset):
    METAINFO = {
        'classes':
        ('corm', 'ground', 'leaf'),
        'palette': [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
                    [255, 0, 255], [100, 100, 255], [200, 200, 100],
                    [170, 120, 200], [255, 0, 0], [200, 100, 100],
                    [10, 200, 100], [200, 200, 200], [50, 50, 50]],
        'seg_valid_class_ids':
        tuple(range(3)),  # 如果你的数据只有3个类别 (0, 1, 2)
        'seg_all_class_ids':
        tuple(range(3))  # 如果你的数据只有3个类别 (0, 1, 2)
    }
