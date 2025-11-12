import torch
import numpy as np
from typing import Optional


class PCFeatureVisualizer:
    """
    Thin adapter to integrate with pointcloud_visualizer-main when present.
    Falls back to no-op if library is not installed.
    """
    def __init__(self, enabled: bool = True, window_title: str = "OneFormer3D Features"):
        self.enabled = enabled
        self.window_title = window_title
        self._backend = None
        self._buffers = []  # list of dicts: {name, points(np[N,3]), colors(np[N,3])}
        if self.enabled:
            try:
                # Lazy import; support common entry names
                from pointcloud_visualizer import Visualizer  # type: ignore
                self._backend = Visualizer(title=self.window_title)
            except Exception:
                # No-op backend
                self._backend = None

    def _to_numpy(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    def _to_rgb(self, scalar: torch.Tensor) -> torch.Tensor:
        """Normalize scalar to [0,1] and convert to simple RGB (blue->red)."""
        s = scalar
        if s.dim() > 1:
            s = s.view(-1)
        s = s.float()
        s = s - torch.min(s)
        denom = torch.max(s)
        if denom > 0:
            s = s / denom
        r = s
        g = 1.0 - torch.abs(s - 0.5) * 2.0
        g = torch.clamp(g, 0.0, 1.0)
        b = 1.0 - s
        rgb = torch.stack([r, g, b], dim=-1)
        return rgb

    def show_scalar(self, points_xyz: torch.Tensor, scalar: torch.Tensor, name: str):
        if not self.enabled:
            return
        if points_xyz is None or scalar is None:
            return
        try:
            pts = points_xyz[..., :3]
            if pts.dim() > 2:
                pts = pts.view(-1, pts.shape[-1])
            if scalar.dim() > 1:
                scalar = scalar.view(-1)
            rgb = self._to_rgb(scalar)
            if self._backend is not None:
                self._backend.add_points(self._to_numpy(pts), colors=self._to_numpy(rgb), name=name)
            else:
                # Graceful fallback: print a brief summary
                print(f"[PCFeatureVisualizer] {name}: points={pts.shape[0]}")
            # Buffer for saving
            self._buffers.append({
                'name': name,
                'points': self._to_numpy(pts),
                'colors': self._to_numpy(rgb)
            })
        except Exception:
            pass

    def show_rgb(self, points_xyz: torch.Tensor, rgb: torch.Tensor, name: str):
        if not self.enabled:
            return
        try:
            pts = points_xyz[..., :3]
            if pts.dim() > 2:
                pts = pts.view(-1, pts.shape[-1])
            if rgb.dim() > 2:
                rgb = rgb.view(-1, rgb.shape[-1])
            if self._backend is not None:
                self._backend.add_points(self._to_numpy(pts), colors=self._to_numpy(rgb), name=name)
            else:
                print(f"[PCFeatureVisualizer] {name}: points={pts.shape[0]}")
            # Buffer for saving
            self._buffers.append({
                'name': name,
                'points': self._to_numpy(pts),
                'colors': self._to_numpy(rgb)
            })
        except Exception:
            pass

    def render(self):
        if not self.enabled:
            return
        try:
            if self._backend is not None:
                self._backend.render()
        except Exception:
            pass

    def save_clouds(self, output_dir: str):
        """Save buffered point clouds to output_dir as ASCII PLY files."""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            for i, item in enumerate(self._buffers):
                name = item.get('name', f'cloud_{i}')
                pts = item['points']
                cols = item['colors']
                # Ensure numpy arrays
                if hasattr(pts, 'shape') and pts.shape[1] >= 3:
                    xyz = pts[:, :3]
                else:
                    continue
                if hasattr(cols, 'shape') and cols.shape[1] >= 3:
                    rgb = (np.clip(cols, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    rgb = np.full_like(xyz, 255, dtype=np.uint8)
                safe_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
                path = os.path.join(output_dir, f"{i:02d}_{safe_name}.ply")
                self._save_ply_ascii(path, xyz, rgb)
        except Exception:
            pass

    def _save_ply_ascii(self, path, xyz, rgb):
        try:
            with open(path, 'w') as f:
                n = xyz.shape[0]
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write(f'element vertex {n}\n')
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
                f.write('end_header\n')
                for i in range(n):
                    x, y, z = xyz[i]
                    r, g, b = rgb[i]
                    f.write(f'{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n')
        except Exception:
            pass


