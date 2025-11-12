import argparse
import os
import sys

from typing import Tuple, Optional
# Ensure repo root and tools dir are on sys.path
_this_file = os.path.abspath(__file__)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(_this_file), '..', '..'))
_tools_dir = os.path.join(_repo_root, 'tools')
for _p in (_repo_root, _tools_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from mmengine.config import Config  # type: ignore
    from mmdet3d.apis import init_model  # type: ignore
    _mm_ok = True
except Exception as _e:
    print(f"[Warn] mmengine/mmdet3d imports failed: {_e}. Will use dummy features if model init is requested.")
    _mm_ok = False
import torch
import numpy as np

from .integrated_cabbage_module import IntegratedCabbageEnhancementModule
from .pc_vis_adapter import PCFeatureVisualizer


def load_points(path: str) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.npy', '.npz']:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if 'points' in arr:
                pts = arr['points']
            else:
                # take first array
                key = list(arr.keys())[0]
                pts = arr[key]
        else:
            pts = arr
    elif ext == '.pcd':
        # Prefer Open3D if available
        try:
            import open3d as o3d  # type: ignore
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points, dtype=np.float32)
        except Exception:
            # Minimal ASCII .pcd parser
            with open(path, 'r') as f:
                header = []
                line = f.readline()
                if not line.startswith('VERSION') and not line.startswith('#'):
                    header.append(line.strip())
                while True:
                    line = f.readline()
                    if not line:
                        break
                    s = line.strip()
                    header.append(s)
                    if s.startswith('DATA'):
                        data_type = s.split()[-1].lower()
                        break
                if data_type != 'ascii':
                    raise RuntimeError('Binary PCD requires open3d; please install open3d to read this file.')
                # Load remaining as ASCII numbers
                data = np.loadtxt(path, comments=None, dtype=np.float32, skiprows=len(header))
                pts = data[:, :3] if data.shape[1] >= 3 else data
    elif ext == '.txt' or ext == '.xyz':
        pts = np.loadtxt(path)
    else:
        # simple PLY fallback without dependencies
        with open(path, 'rb') as f:
            header = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header.append(line)
                if line.startswith('end_header'):
                    break
            content = f.read()
        # naive float32 triplets (this assumes binary little-endian float32 x y z)
        pts = np.frombuffer(content, dtype=np.float32)
        pts = pts.reshape(-1, 3)
    if pts.shape[-1] > 3:
        pts = pts[:, :3]
    pts = pts.astype(np.float32)
    pts = torch.from_numpy(pts).unsqueeze(0)  # [1, N, 3]
    return pts


def dummy_backbone(points: torch.Tensor, feature_dim: int, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Placeholder to produce shapes if user does not provide a model.
    Recommend replacing with your actual model forward that returns (features, logits).
    """
    b, n, _ = points.shape
    device = points.device
    torch.manual_seed(0)
    features = torch.randn(b, n, feature_dim, device=device)
    logits = torch.randn(b, n, num_classes, device=device)
    return features, logits


def main():
    parser = argparse.ArgumentParser(description='Point cloud feature visualization for cabbage modules')
    parser.add_argument('--pc', required=True, help='Path to point cloud file (.npy/.npz/.txt/.xyz/.ply)')
    parser.add_argument('--ckpt', default=None, help='Optional checkpoint for your backbone model')
    parser.add_argument('--config', default=None, help='Optional config file to build model (MMEngine config)')
    parser.add_argument('--feature-dim', type=int, default=256, help='Backbone feature dim for OpenFormer3D features')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of segmentation classes')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--window-title', default='Cabbage Feature Viz', help='Visualizer window title')
    parser.add_argument('--save-dir', default=None, help='If set, saves colored point clouds to this directory')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering (debug print only)')
    parser.add_argument('--viz-sem-class', default=None, help='Comma-separated semantic class indices to visualize from logits, e.g., "0,2"')
    parser.add_argument('--save-npz-dir', default=None, help='Also save NPZ features for head/interact/leaf stages')
    args = parser.parse_args()

    # Normalize device selection for both PyTorch and MMDet3D init
    requested_device = args.device.lower()
    if requested_device.startswith('cuda') and torch.cuda.is_available():
        torch_device = torch.device(requested_device)
        mm_device_str = requested_device if ':' in requested_device else 'cuda:0'
    else:
        torch_device = torch.device('cpu')
        mm_device_str = 'cpu'

    # 1) Load points
    points = load_points(args.pc).to(torch_device)

    # 2) TODO: Replace this with your backbone forward that returns (features, logits)
    #    If you have an existing model, load it here and call model(points)
    #    For example:
    #    model = YourModel(...).to(device)
    #    state = torch.load(args.ckpt, map_location=device)
    #    model.load_state_dict(state.get('state_dict', state), strict=False)
    #    model.eval()
    #    with torch.no_grad():
    #        openformer_features, openformer_logits = model(points)
    with torch.no_grad():
        # Prefer loading model from config if provided
        if args.config is not None and args.ckpt is not None:
            try:
                cfg = Config.fromfile(args.config)
                # MMDet3D expects a string device like 'cpu' or 'cuda:0'
                model = init_model(cfg, args.ckpt, device=mm_device_str)
                # Ensure model and inputs are on the same device
                model = model.to(torch_device)
                model.eval()
                # 优先尝试 forward_features 接口
                openformer_features, openformer_logits = None, None
                if hasattr(model, 'forward_features'):
                    try:
                        out = model.forward_features(points)
                        if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                            openformer_features, openformer_logits = out[0], out[1]
                    except Exception:
                        pass
                # 次之尝试直接 forward
                if openformer_features is None or openformer_logits is None:
                    out = model(points)
                    if isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                        openformer_features, openformer_logits = out[0], out[1]
                # 仍失败则回退
                if openformer_features is None or openformer_logits is None:
                    print('[Warn] Model did not return (features, logits); using dummy features/logits.')
                    openformer_features, openformer_logits = dummy_backbone(points, args.feature_dim, args.num_classes)
            except Exception as e:
                print(f"[Warn] Failed to init/use config model: {e}; using dummy features/logits.")
                openformer_features, openformer_logits = dummy_backbone(points, args.feature_dim, args.num_classes)
        else:
            openformer_features, openformer_logits = dummy_backbone(points, args.feature_dim, args.num_classes)
    # 3) Create visualizer and integrated enhancer
    vis = PCFeatureVisualizer(enabled=not args.no_render, window_title=args.window_title)
    enhancer = IntegratedCabbageEnhancementModule(
        openformer_feature_dim=args.feature_dim,
        visualizer=vis
    ).to(torch_device).eval()

    # 4) Run forward to visualize head/leaf/interaction effects
    with torch.no_grad():
        # Optional: visualize selected semantic class probabilities from original logits
        if args.viz_sem_class is not None:
            try:
                class_indices = [int(x) for x in str(args.viz_sem_class).split(',') if x.strip() != '']
                if torch.is_tensor(openformer_logits) and openformer_logits.dim() == 3:
                    probs0 = torch.softmax(openformer_logits[0], dim=-1)
                    for ci in class_indices:
                        if 0 <= ci < probs0.shape[-1]:
                            vis.show_scalar(points[0], probs0[:, ci], name=f"SemClassProb_{ci}")
            except Exception:
                pass
        outputs = enhancer(points, openformer_features, openformer_logits)

    # 5) Save outputs if requested
    if args.save_dir is not None:
        try:
            vis.save_clouds(args.save_dir)
            print(f"Saved visualizations to: {args.save_dir}")
        except Exception as e:
            print(f"Failed to save visualizations: {e}")

    # 5.1) Save NPZ features for three stages if requested
    if args.save_npz_dir is not None:
        try:
            from .features_vis_save import feature_vis
            saver = feature_vis(save_root=args.save_npz_dir)
            if isinstance(outputs, dict):
                if 'features_after_head_refinement' in outputs:
                    saver.feature_save(points, outputs['features_after_head_refinement'], 'head_refine_feat')
                if 'features_after_interaction' in outputs:
                    saver.feature_save(points, outputs['features_after_interaction'], 'interaction_feat')
                if 'features_after_leaf_enhancement' in outputs:
                    saver.feature_save(points, outputs['features_after_leaf_enhancement'], 'leaf_enhance_feat')
            print(f"Saved NPZ features to: {os.path.abspath(args.save_npz_dir)}")
        except Exception as e:
            print(f"[Warn] Failed to save NPZ features: {e}")


if __name__ == '__main__':
    main()


