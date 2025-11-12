import argparse
import os
import sys
from typing import Optional

import numpy as np


def _load_npz(path: str):
    data = np.load(path)
    xyz = data['xyz']
    features = data['features']
    layer_name = str(data['layer_name']) if 'layer_name' in data else 'layer'
    return xyz, features, layer_name


def _normalize_to_rgb(feats: np.ndarray) -> np.ndarray:
    """Map features [N,C] to RGB [N,3]. Default: max-channel coloring.
    If C>=3 and user wants the first 3 channels, this can be extended via args.
    """
    N, C = feats.shape
    if C == 1:
        s = feats[:, 0]
        s = s - s.min()
        denom = s.max() if s.max() > 0 else 1.0
        s = s / denom
        r = s
        g = 1.0 - np.abs(s - 0.5) * 2.0
        g = np.clip(g, 0.0, 1.0)
        b = 1.0 - s
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)

    # Max-channel coloring
    idx = np.argmax(feats, axis=1)
    # Simple LUT for up to 10 channels; repeats if more
    lut = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [255, 128, 0], [128, 0, 255], [0, 128, 255], [128, 128, 128]
    ], dtype=np.uint8)
    rgb = lut[idx % len(lut)]
    return rgb


def _show_open3d(xyz: np.ndarray, rgb: np.ndarray, window: str = 'FeatureViz'):
    try:
        import open3d as o3d
    except Exception as e:
        print('Open3D is required for visualization. Please install via: pip install open3d')
        sys.exit(1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float32) / 255.0))
    o3d.visualization.draw_geometries([pcd], window_name=window)


def resolve_file(args) -> Optional[str]:
    if args.file is not None:
        return args.file
    if args.layer_name is not None:
        # Read latest index file written by feature_vis
        data_dir = args.data_dir
        index_path = os.path.join(data_dir, f'{args.layer_name}_latest.txt')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        # Fallback: find most recent .npz with layer prefix
        prefix = f'{args.layer_name}_'
        candidates = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir) if fn.endswith('.npz') and fn.startswith(prefix)]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description='Visualize saved point features')
    parser.add_argument('--file', type=str, default=None, help='Path to saved .npz file')
    parser.add_argument('--layer-name', type=str, default=None, help='Layer name (uses latest saved for that layer)')
    parser.add_argument('--data-dir', type=str, default='/home/lihongda/3D/oneformer3d/npz_out/', help='Directory where .npz files are saved')
    parser.add_argument('--window', type=str, default='Feature Visualization', help='Open3D window title')
    parser.add_argument('--args', action='store_true', help='Placeholder to match your interface')
    args = parser.parse_args()

    path = resolve_file(args)
    if path is None or not os.path.exists(path):
        print('No .npz file found. Use --file or --layer-name, and ensure features are saved first.')
        sys.exit(1)

    xyz, features, layer_name = _load_npz(path)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    rgb = _normalize_to_rgb(features)
    _show_open3d(xyz, rgb, window=args.window or layer_name)


if __name__ == '__main__':
    main()


