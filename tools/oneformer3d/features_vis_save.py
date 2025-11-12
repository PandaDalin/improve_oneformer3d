import os
import time
from typing import Optional

import numpy as np
import torch


class feature_vis:
    """
    Save point coordinates (xyz) and per-point features for later visualization.

    Usage:
        from features_vis_save import feature_vis
        self.Visualize_feature = feature_vis(save_root='./visualization_tool/data')
        self.Visualize_feature.feature_save(xyz=li_xyz, features=li_features, layer_name=layer_name)
    """

    def __init__(self, save_root: str = './visualization_tool/data'):
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=True)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def feature_save(self,
                     xyz: torch.Tensor,
                     features: torch.Tensor,
                     layer_name: str,
                     filename: Optional[str] = None) -> str:
        """
        Save xyz and features into an .npz file under save_root.

        - xyz: [N,3] or [B,N,3]
        - features: [N,C] or [B,N,C]
        - layer_name: identifier used in filename
        - filename: optional explicit filename (without directory)

        Returns: absolute path to saved .npz
        """
        # Ensure numpy arrays with shapes [N,3] and [N,C]
        pts = self._to_numpy(xyz)
        feats = self._to_numpy(features)

        if pts.ndim == 3:
            # take batch 0 by default
            pts = pts[0]
        if feats.ndim == 3:
            feats = feats[0]

        if pts.shape[-1] > 3:
            pts = pts[:, :3]

        # Build filename
        if filename is None:
            ts = time.strftime('%Y%m%d_%H%M%S')
            base = f'{layer_name}_{ts}.npz'
        else:
            base = filename if filename.endswith('.npz') else f'{filename}.npz'

        path = os.path.join(self.save_root, base)

        # Save
        np.savez_compressed(path, xyz=pts.astype(np.float32), features=feats.astype(np.float32), layer_name=layer_name)
        # Also write a small index file with latest path for this layer
        try:
            index_path = os.path.join(self.save_root, f'{layer_name}_latest.txt')
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(os.path.abspath(path))
        except Exception:
            pass

        return os.path.abspath(path)


