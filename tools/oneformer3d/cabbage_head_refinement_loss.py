import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import math

from mmdet3d.registry import MODELS


class GeometricConstraints(nn.Module):
    """几何约束计算器"""
    def __init__(self):
        super().__init__()
        
    def ellipsoid_shape_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        椭球形状约束损失 - 球径应该接近椭球形
        Args:
            points: [N, 3] 球径区域的点
        Returns:
            loss: scalar 椭球形状偏差
        """
        if len(points) < 10:
            return torch.tensor(0.0, device=points.device)
        
        # 计算点云中心
        center = torch.mean(points, dim=0)  # [3]
        centered_points = points - center  # [N, 3]
        
        # 计算协方差矩阵
        cov_matrix = torch.mm(centered_points.T, centered_points) / len(points)
        
        try:
            # 特征值分解
            eigenvalues, _ = torch.linalg.eigh(cov_matrix)
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]
            
            # 椭球的长半轴比例应该在合理范围内
            # 甘蓝球径通常接近球形，长短轴比例不应该太大
            a, b, c = eigenvalues[0], eigenvalues[1], eigenvalues[2]
            
            # 理想情况下 a ≈ b ≈ c (球形)
            ideal_ratio_ab = 1.0
            ideal_ratio_ac = 1.0
            
            actual_ratio_ab = b / (a + 1e-8)
            actual_ratio_ac = c / (a + 1e-8)
            
            # 损失 - 偏离球形的程度
            shape_loss = (F.mse_loss(actual_ratio_ab, torch.tensor(ideal_ratio_ab, device=points.device)) +
                         F.mse_loss(actual_ratio_ac, torch.tensor(ideal_ratio_ac, device=points.device)))
            
            return shape_loss
            
        except:
            return torch.tensor(0.0, device=points.device)
    
    def size_consistency_loss(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        尺寸一致性损失 - 预测球径大小应该与真值接近
        Args:
            pred_mask: [N] 预测球径掩码
            gt_mask: [N] 真值球径掩码
        Returns:
            loss: scalar
        """
        pred_volume = torch.sum(pred_mask.float())
        gt_volume = torch.sum(gt_mask.float())
        
        # 体积差异损失
        volume_loss = F.mse_loss(pred_volume, gt_volume)
        
        # 相对误差
        if gt_volume > 0:
            relative_error = torch.abs(pred_volume - gt_volume) / gt_volume
            return volume_loss + 0.5 * relative_error
        else:
            return volume_loss
    
    def surface_smoothness_loss(self, 
                               points: torch.Tensor, 
                               boundary_mask: torch.Tensor,
                               seg_probs: torch.Tensor) -> torch.Tensor:
        """
        表面平滑性损失 - 球径表面应该平滑
        Args:
            points: [N, 3] 所有点
            boundary_mask: [N] 边界点掩码
            seg_probs: [N, 3] 分割概率
        Returns:
            loss: scalar
        """
        if torch.sum(boundary_mask) < 5:
            return torch.tensor(0.0, device=points.device)
        
        boundary_points = points[boundary_mask]  # [M, 3]
        boundary_probs = seg_probs[boundary_mask]  # [M, 3]
        
        # 计算边界点的局部平滑性
        smoothness_losses = []
        
        for i in range(len(boundary_points)):
            center_point = boundary_points[i]
            center_prob = boundary_probs[i, 2]  # 边界概率
            
            # 找到邻近的边界点
            distances = torch.norm(boundary_points - center_point, dim=1)
            close_mask = distances < 0.05  # 5cm邻域
            
            if torch.sum(close_mask) > 1:
                neighbor_probs = boundary_probs[close_mask, 2]
                # 局部概率应该相似 (平滑性)
                prob_variance = torch.var(neighbor_probs)
                smoothness_losses.append(prob_variance)
        
        if len(smoothness_losses) > 0:
            return torch.stack(smoothness_losses).mean()
        else:
            return torch.tensor(0.0, device=points.device)
    
    def connectivity_loss(self, points: torch.Tensor, head_mask: torch.Tensor) -> torch.Tensor:
        """
        连通性约束损失 - 球径区域应该是连通的
        Args:
            points: [N, 3] 所有点
            head_mask: [N] 球径区域掩码
        Returns:
            loss: scalar
        """
        if torch.sum(head_mask) < 5:
            return torch.tensor(0.0, device=points.device)
        
        head_points = points[head_mask]
        
        # 简化的连通性检查：计算点云的紧密度
        center = torch.mean(head_points, dim=0)
        distances = torch.norm(head_points - center, dim=1)
        
        # 连通区域的点应该相对集中
        distance_variance = torch.var(distances)
        max_distance = torch.max(distances)
        
        # 如果点过于分散，增加损失
        connectivity_loss = distance_variance / (max_distance + 1e-8)
        
        return connectivity_loss


@MODELS.register_module()
class CabbageHeadRefinementLoss(nn.Module):
    """
    OpenFormer3D球径分割精度增强损失函数
    专注于提升球径分割精度，不改变原有分割类别
    """
    def __init__(self, 
                 refinement_weight: float = 0.3,
                 consistency_weight: float = 0.2,
                 boundary_weight: float = 2.0,
                 shape_weight: float = 0.5,
                 smoothness_weight: float = 0.3,
                 size_weight: float = 0.8,
                 connectivity_weight: float = 0.6):
        super().__init__()
        
        self.refinement_weight = refinement_weight
        self.consistency_weight = consistency_weight
        self.boundary_weight = boundary_weight
        self.shape_weight = shape_weight
        self.smoothness_weight = smoothness_weight
        self.size_weight = size_weight
        self.connectivity_weight = connectivity_weight
        
        # 基础损失函数
        self.seg_criterion = nn.CrossEntropyLoss(reduction='none')
        
        # 几何约束计算器
        self.geometric_constraints = GeometricConstraints()
        
    def forward(self, 
                enhanced_outputs: Dict[str, torch.Tensor],
                targets, 
                points,
                stage_probs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算增强模块的损失
        Args:
            enhanced_outputs: dict包含增强模块的输出
                - logits: [B, N, C] 精细化的分割logits
                - original_logits: [B, N, C] 原始OpenFormer3D logits  
                - head_mask_prob: [B, N] 球径区域概率
            targets: [B, N] 分割真值标签
            points: [B, N, 3] 点云坐标
            stage_probs: [B, 3] 生长期概率 (可选)
        Returns:
            losses: dict包含各项损失
        """
        try:
            # 获取设备信息
            if 'logits' in enhanced_outputs:
                if isinstance(enhanced_outputs['logits'], list) and len(enhanced_outputs['logits']) > 0:
                    device = enhanced_outputs['logits'][0].device
                else:
                    device = enhanced_outputs['logits'].device
            else:
                device = 'cpu'
            
            # 初始化损失
            total_loss = torch.tensor(0.0, device=device)
            refinement_loss = torch.tensor(0.0, device=device)
            consistency_loss = torch.tensor(0.0, device=device)
            geometric_loss = torch.tensor(0.0, device=device)
            
            # 检查是否有有效的增强输出
            if 'logits' not in enhanced_outputs or 'original_logits' not in enhanced_outputs:
                print("Warning: Missing required enhanced outputs for loss computation")
                return {
                    'total_loss': total_loss,
                    'refinement_loss': refinement_loss,
                    'consistency_loss': consistency_loss,
                    'geometric_loss': geometric_loss
                }
            
            # 处理批次数据：将可能的 [B, N, *] 张量拆分为按样本的列表
            def to_per_sample_list(t):
                if isinstance(t, list):
                    return t
                if not torch.is_tensor(t):
                    return []
                if t.dim() >= 3:
                    # 假设第0维是batch维
                    return [t[i] for i in range(t.shape[0])]
                else:
                    return [t]

            logits_list = to_per_sample_list(enhanced_outputs['logits'])
            original_logits_list = to_per_sample_list(enhanced_outputs['original_logits'])
            head_mask_prob_list = to_per_sample_list(enhanced_outputs.get('head_mask_prob', []))
            
            # 确保targets和points是按样本的列表
            if not isinstance(targets, list):
                targets = to_per_sample_list(targets)
            if not isinstance(points, list):
                points = to_per_sample_list(points)
            
            batch_size = len(logits_list)
            
            for b in range(batch_size):
                if b >= len(targets) or b >= len(points):
                    continue
                    
                logits_b = logits_list[b]
                orig_logits_b = original_logits_list[b]
                targets_b = targets[b]
                points_b = points[b]

                # 规范化形状：去掉前导的1批次维度，并确保为 [N, C] / [N, 3] / [N]
                if torch.is_tensor(logits_b) and logits_b.dim() == 3 and logits_b.shape[0] == 1:
                    logits_b = logits_b.squeeze(0)
                if torch.is_tensor(orig_logits_b) and orig_logits_b.dim() == 3 and orig_logits_b.shape[0] == 1:
                    orig_logits_b = orig_logits_b.squeeze(0)
                if torch.is_tensor(points_b) and points_b.dim() == 3 and points_b.shape[0] == 1:
                    points_b = points_b.squeeze(0)
                if torch.is_tensor(targets_b) and targets_b.dim() == 2 and targets_b.shape[0] == 1:
                    targets_b = targets_b.squeeze(0)
                
                # 获取球径掩码概率
                if b < len(head_mask_prob_list):
                    head_prob_b = head_mask_prob_list[b]
                else:
                    head_prob_b = torch.zeros(logits_b.shape[0], device=device)

                if torch.is_tensor(head_prob_b) and head_prob_b.dim() == 2 and head_prob_b.shape[0] == 1:
                    head_prob_b = head_prob_b.squeeze(0)
                
                # 对齐长度，避免长度不一致
                try:
                    lengths = []
                    if torch.is_tensor(logits_b):
                        lengths.append(logits_b.shape[0])
                    if torch.is_tensor(orig_logits_b):
                        lengths.append(orig_logits_b.shape[0])
                    if torch.is_tensor(points_b):
                        lengths.append(points_b.shape[0])
                    if torch.is_tensor(targets_b):
                        lengths.append(targets_b.shape[0])
                    if torch.is_tensor(head_prob_b):
                        lengths.append(head_prob_b.shape[0])
                    if len(lengths) > 0:
                        min_len = min(lengths)
                        if torch.is_tensor(logits_b) and logits_b.shape[0] != min_len:
                            logits_b = logits_b[:min_len]
                        if torch.is_tensor(orig_logits_b) and orig_logits_b.shape[0] != min_len:
                            orig_logits_b = orig_logits_b[:min_len]
                        if torch.is_tensor(points_b) and points_b.shape[0] != min_len:
                            points_b = points_b[:min_len]
                        if torch.is_tensor(targets_b) and targets_b.shape[0] != min_len:
                            targets_b = targets_b[:min_len]
                        if torch.is_tensor(head_prob_b) and head_prob_b.shape[0] != min_len:
                            head_prob_b = head_prob_b[:min_len]
                except Exception as e:
                    print(f"Warning: Failed to align lengths: {e}")

                # 1. 精细化损失 - 使用交叉熵
                if logits_b.numel() > 0 and targets_b.numel() > 0:
                    # 确保标签在有效范围内
                    num_classes = logits_b.shape[-1]
                    valid_targets = torch.clamp(targets_b, 0, num_classes - 1)
                    
                    # 计算交叉熵损失
                    ce_loss = self.seg_criterion(logits_b, valid_targets)
                    refinement_loss += ce_loss.mean()
                
                # 2. 一致性损失 - KL散度
                if logits_b.numel() > 0 and orig_logits_b.numel() > 0:
                    try:
                        kl_loss = F.kl_div(
                            F.log_softmax(logits_b, dim=-1),
                            F.softmax(orig_logits_b, dim=-1),
                            reduction='batchmean'
                        )
                        consistency_loss += kl_loss
                    except Exception as e:
                        print(f"Error computing KL divergence: {e}")
                
                # 3. 几何约束损失
                if points_b.numel() > 0 and head_prob_b.numel() > 0:
                    try:
                        # 获取球径预测掩码
                        head_pred_mask = head_prob_b > 0.5
                        
                        # 获取球径真值掩码
                        head_gt_mask = self._get_head_class_mask(targets_b)
                        # 对齐掩码长度
                        if head_gt_mask.numel() != head_pred_mask.numel():
                            min_len = min(head_gt_mask.numel(), head_pred_mask.numel())
                            head_gt_mask = head_gt_mask[:min_len]
                            head_pred_mask = head_pred_mask[:min_len]
                        
                        # 计算几何约束损失
                        geo_loss = self._compute_head_geometric_loss(
                            points_b, logits_b, targets_b, head_prob_b
                        )
                        geometric_loss += geo_loss
                    except Exception as e:
                        print(f"Error computing geometric loss: {e}")
            
            # 计算总损失
            total_loss = (self.refinement_weight * refinement_loss + 
                         self.consistency_weight * consistency_loss + 
                         geometric_loss)  # geometric_loss already includes weights
            
            return {
                'total_loss': total_loss,
                'refinement_loss': refinement_loss,
                'consistency_loss': consistency_loss,
                'geometric_loss': geometric_loss
            }
            
        except Exception as e:
            print(f"Error in forward method: {e}")
            # 返回零损失作为fallback
            try:
                device = enhanced_outputs.get('logits', [torch.tensor(0.0)])[0].device if isinstance(enhanced_outputs.get('logits', []), list) and len(enhanced_outputs.get('logits', [])) > 0 else 'cpu'
            except:
                device = 'cpu'
            
            return {
                'total_loss': torch.tensor(0.0, device=device),
                'refinement_loss': torch.tensor(0.0, device=device),
                'consistency_loss': torch.tensor(0.0, device=device),
                'geometric_loss': torch.tensor(0.0, device=device)
            }
    
    def _get_head_class_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """
        获取球径类别掩码 (根据你的具体类别定义调整)
        假设球径类别的标签值为特定值，比如标签1代表球径
        """
        # 安全检查
        if not torch.is_tensor(targets) or targets.dim() == 0:
            print(f"Warning: Invalid targets in _get_head_class_mask. Returning empty mask.")
            # 安全地获取设备信息
            try:
                device = targets.device if torch.is_tensor(targets) else 'cpu'
            except:
                device = 'cpu'
            return torch.zeros(1, device=device, dtype=torch.float)
        
        # 数据集类别: 0=corm(球径), 1=ground, 2=leaf
        # 球径应对应类别0
        head_mask = (targets == 0)
        return head_mask.float()
    
    def _compute_head_geometric_loss(self,
                                    points: torch.Tensor,
                                    logits: torch.Tensor, 
                                    targets: torch.Tensor,
                                    head_prob: torch.Tensor) -> torch.Tensor:
        """
        计算球径几何约束损失
        """
        # 获取预测的球径区域
        pred_probs = F.softmax(logits, dim=-1)
        head_pred_mask = head_prob > 0.5
        
        # 安全地处理targets
        if not torch.is_tensor(targets) or targets.dim() == 0:
            print(f"Warning: Invalid targets in geometric loss computation. Using default mask.")
            head_gt_mask = torch.zeros_like(head_pred_mask, dtype=torch.bool)
        else:
            try:
                head_gt_mask = self._get_head_class_mask(targets.unsqueeze(0)).squeeze(0).bool()
            except Exception as e:
                print(f"Error computing head gt mask: {e}. Using default mask.")
                head_gt_mask = torch.zeros_like(head_pred_mask, dtype=torch.bool)
        
        losses = []
        
        # 1. 球径形状约束
        if torch.sum(head_pred_mask) > 10:
            try:
                # 安全地获取点云数据
                masked_points = points[head_pred_mask]
                if masked_points.numel() > 0 and not torch.isnan(masked_points).any():
                    shape_loss = self.geometric_constraints.ellipsoid_shape_loss(masked_points)
                    losses.append(self.shape_weight * shape_loss)
                else:
                    print(f"Warning: Invalid masked points for shape loss computation.")
            except Exception as e:
                print(f"Error computing ellipsoid shape loss: {e}. Skipping this loss.")
        
        # 2. 尺寸一致性
        if torch.sum(head_gt_mask) > 0:
            try:
                # 确保掩码是有效的
                if head_pred_mask.numel() == head_gt_mask.numel():
                    size_loss = self.geometric_constraints.size_consistency_loss(
                        head_pred_mask, head_gt_mask
                    )
                    losses.append(self.size_weight * size_loss)
                else:
                    print(f"Warning: Mask size mismatch for size consistency loss.")
            except Exception as e:
                print(f"Error computing size consistency loss: {e}. Skipping this loss.")
        
        # 3. 表面平滑性损失
        # S3DIS甘蓝设置中没有单独的边界类别，这里用预测的球径边界近似：
        # 使用 head_prob 在(0.4, 0.6)之间的点作为近似边界
        if torch.is_tensor(head_prob) and head_prob.numel() == points.shape[0]:
            boundary_mask = (head_prob > 0.4) & (head_prob < 0.6)
        else:
            boundary_mask = torch.zeros_like(head_pred_mask, dtype=torch.bool)
            
        if torch.sum(boundary_mask) > 5:
            try:
                # 确保所有输入都是有效的
                if (points.numel() > 0 and boundary_mask.numel() > 0 and 
                    pred_probs.numel() > 0 and not torch.isnan(points).any()):
                    smoothness_loss = self.geometric_constraints.surface_smoothness_loss(
                        points, boundary_mask, pred_probs
                    )
                    losses.append(self.smoothness_weight * smoothness_loss)
                else:
                    print(f"Warning: Invalid inputs for surface smoothness loss.")
            except Exception as e:
                print(f"Error computing surface smoothness loss: {e}. Skipping this loss.")
        
        # 4. 连通性约束
        if torch.sum(head_pred_mask) > 5:
            try:
                # 确保点云数据有效
                if points.numel() > 0 and not torch.isnan(points).any():
                    connectivity_loss = self.geometric_constraints.connectivity_loss(
                        points, head_pred_mask
                    )
                    losses.append(self.connectivity_weight * connectivity_loss)
                else:
                    print(f"Warning: Invalid points for connectivity loss.")
            except Exception as e:
                print(f"Error computing connectivity loss: {e}. Skipping this loss.")
        
        if len(losses) > 0:
            try:
                # 确保所有损失都是有效的张量
                valid_losses = [loss for loss in losses if torch.is_tensor(loss) and loss.numel() > 0]
                if valid_losses:
                    return torch.stack(valid_losses).sum()
                else:
                    print(f"Warning: No valid losses to stack.")
                    return torch.tensor(0.0, device=points.device)
            except Exception as e:
                print(f"Error stacking geometric losses: {e}. Using default value.")
                return torch.tensor(0.0, device=points.device)
        else:
            return torch.tensor(0.0, device=points.device)


class BoundaryDetector(nn.Module):
    """边界检测辅助函数"""
    def __init__(self, threshold: float = 0.02):
        super().__init__()
        self.threshold = threshold
        
    def detect_boundary_regions(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        从分割标签中检测边界区域
        Args:
            segmentation: [N] 分割标签
        Returns:
            boundary_mask: [N] 边界掩码
        """
        # 简化实现：边界点标签为2
        try:
            if torch.is_tensor(segmentation) and segmentation.dim() > 0:
                boundary_mask = (segmentation == 2)
            else:
                boundary_mask = torch.zeros(1, device='cpu', dtype=torch.bool)
            return boundary_mask
        except Exception as e:
            print(f"Error detecting boundary regions: {e}. Using default mask.")
            return torch.zeros(1, device='cpu', dtype=torch.bool)


class AdaptiveLossWeighting(nn.Module):
    """自适应损失权重调整"""
    def __init__(self, initial_weights: Dict[str, float]):
        super().__init__()
        self.weights = nn.ParameterDict({
            key: nn.Parameter(torch.tensor(weight, dtype=torch.float32))
            for key, weight in initial_weights.items()
        })
        
    def forward(self, losses: Dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        """
        自适应调整各项损失的权重
        Args:
            losses: dict包含各项损失
            epoch: 当前epoch
        Returns:
            weighted_loss: 加权后的总损失
        """
        # 温度退火：随着训练进行，几何约束权重逐渐增加
        try:
            geometric_weight = self.weights['geometric'] * min(1.0, epoch / 50.0)
            
            weighted_loss = (
                self.weights['segmentation'] * losses['segmentation_loss'] +
                self.weights['boundary'] * losses['boundary_loss'] +
                geometric_weight * losses['geometric_loss']
            )
        except Exception as e:
            print(f"Error computing weighted loss: {e}. Using default weights.")
            weighted_loss = losses.get('total_loss', torch.tensor(0.0))
        
        try:
            return weighted_loss
        except Exception as e:
            print(f"Error returning weighted loss: {e}. Using default value.")
            return torch.tensor(0.0)


@MODELS.register_module()
class CabbageHeadRefinementCriterion:
    """
    甘蓝球径分割增强损失函数 - 与OneFormer3D框架集成
    """
    def __init__(self,
                 refinement_weight: float = 0.3,
                 consistency_weight: float = 0.2,
                 boundary_weight: float = 2.0,
                 shape_weight: float = 0.5,
                 smoothness_weight: float = 0.3,
                 size_weight: float = 0.8,
                 connectivity_weight: float = 0.6,
                 loss_weight: float = 1.0):
        self.refinement_loss = CabbageHeadRefinementLoss(
            refinement_weight=refinement_weight,
            consistency_weight=consistency_weight,
            boundary_weight=boundary_weight,
            shape_weight=shape_weight,
            smoothness_weight=smoothness_weight,
            size_weight=size_weight,
            connectivity_weight=connectivity_weight
        )
        self.loss_weight = loss_weight
    
    def __call__(self, pred, insts, points=None, stage_probs=None):
        """
        计算增强损失
        Args:
            pred: 预测结果字典
            insts: 真值实例列表
            points: 点云坐标 [B, N, 3]
            stage_probs: 生长期概率 [B, 3]
        Returns:
            loss_dict: 损失字典
        """
        # 这里需要根据实际的pred和insts格式进行调整
        # 假设pred包含增强模块的输出
        try:
            if 'enhanced_outputs' not in pred:
                # 如果没有增强输出，返回零损失
                try:
                    device = pred['masks'][0].device if 'masks' in pred and len(pred['masks']) > 0 else 'cpu'
                except Exception as e:
                    print(f"Error getting device from masks: {e}. Using cpu.")
                    device = 'cpu'
                return {
                    'cabbage_total_loss': torch.tensor(0.0, device=device),
                    'cabbage_refinement_loss': torch.tensor(0.0, device=device),
                    'cabbage_consistency_loss': torch.tensor(0.0, device=device),
                    'cabbage_geometric_loss': torch.tensor(0.0, device=device)
                }
            
            enhanced_outputs = pred['enhanced_outputs']
        except Exception as e:
            print(f"Error checking enhanced outputs: {e}. Using default losses.")
            try:
                device = pred.get('masks', [torch.tensor(0.0)])[0].device if 'masks' in pred and len(pred['masks']) > 0 else 'cpu'
            except Exception as e:
                print(f"Error getting device from pred: {e}. Using cpu.")
                device = 'cpu'
            return {
                'cabbage_total_loss': torch.tensor(0.0, device=device),
                'cabbage_refinement_loss': torch.tensor(0.0, device=device),
                'cabbage_consistency_loss': torch.tensor(0.0, device=device),
                'cabbage_geometric_loss': torch.tensor(0.0, device=device)
            }
        
        # 提取真值标签（使用语义superpoint掩码，忽略背景行）
        targets = []
        try:
            for inst in insts:
                if hasattr(inst, 'sp_sem_masks') and inst.sp_sem_masks is not None:
                    sem_masks = inst.sp_sem_masks  # [n_classes+1, N_i]
                    n_rows = sem_masks.shape[0]
                    if n_rows >= 2:
                        # 使用前 n_rows-1 行（排除背景）
                        sem_no_bg = sem_masks[: n_rows - 1, :]
                    else:
                        sem_no_bg = sem_masks
                    target = sem_no_bg.float().argmax(0)
                    targets.append(target)
                else:
                    # 若缺失，回退到全零标签
                    try:
                        device = pred['masks'][0].device
                        length = pred['masks'][0].shape[1]
                    except Exception as e:
                        print(f"Error getting device/length: {e}. Using defaults.")
                        device = 'cpu'
                        length = 1
                    targets.append(torch.zeros(length, dtype=torch.long, device=device))
        except Exception as e:
            print(f"Error extracting targets: {e}. Using default targets.")
            # 创建默认目标
            try:
                device = pred['masks'][0].device
                length = pred['masks'][0].shape[1]
            except Exception as e2:
                print(f"Error getting device/length for default targets: {e2}. Using defaults.")
                device = 'cpu'
                length = 1
            targets = [torch.zeros(length, dtype=torch.long, device=device)]
        
        # 计算损失
        # 不堆叠，保持list以支持可变点数
        try:
            losses = self.refinement_loss(enhanced_outputs, targets, points, stage_probs)
        except Exception as e:
            print(f"Error computing refinement loss: {e}. Using default losses.")
            try:
                device = pred['masks'][0].device
            except Exception as e2:
                print(f"Error getting device for default losses: {e2}. Using cpu.")
                device = 'cpu'
            losses = {
                'total_loss': torch.tensor(0.0, device=device),
                'refinement_loss': torch.tensor(0.0, device=device),
                'consistency_loss': torch.tensor(0.0, device=device),
                'geometric_loss': torch.tensor(0.0, device=device)
            }
        
        # 应用权重
        weighted_losses = {}
        try:
            for key, value in losses.items():
                weighted_losses[f'cabbage_{key}'] = self.loss_weight * value
        except Exception as e:
            print(f"Error applying loss weights: {e}. Using default weighted losses.")
            try:
                device = pred['masks'][0].device
            except Exception as e2:
                print(f"Error getting device for default weighted losses: {e2}. Using cpu.")
                device = 'cpu'
            weighted_losses = {
                'cabbage_total_loss': torch.tensor(0.0, device=device),
                'cabbage_refinement_loss': torch.tensor(0.0, device=device),
                'cabbage_consistency_loss': torch.tensor(0.0, device=device),
                'cabbage_geometric_loss': torch.tensor(0.0, device=device)
            }
        
        try:
            return weighted_losses
        except Exception as e:
            print(f"Error returning weighted losses: {e}. Using default losses.")
            try:
                device = pred['masks'][0].device
            except Exception as e2:
                print(f"Error getting device for final return: {e2}. Using cpu.")
                device = 'cpu'
            return {
                'cabbage_total_loss': torch.tensor(0.0, device=device),
                'cabbage_refinement_loss': torch.tensor(0.0, device=device),
                'cabbage_consistency_loss': torch.tensor(0.0, device=device),
                'cabbage_geometric_loss': torch.tensor(0.0, device=device)
            }
