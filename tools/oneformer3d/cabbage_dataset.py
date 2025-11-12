from mmdet3d.registry import DATASETS



@DATASETS.register_module()
class CabbageSegDataset_():
    METAINFO = {
        'classes':
        ('corm', 'ground', 'leaf'),
        'palette': [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
                    [255, 0, 255], [100, 100, 255], [200, 200, 100],
                    [170, 120, 200], [255, 0, 0], [200, 100, 100],
                    [10, 200, 100], [200, 200, 200], [50, 50, 50]],
        'seg_valid_class_ids':
        tuple(range(3)),
        'seg_all_class_ids':
        tuple(range(3))  # possibly with 'stair' class
    }