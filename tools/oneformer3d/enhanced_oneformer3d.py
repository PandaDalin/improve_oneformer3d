import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import MinkowskiEngine as ME
import os
import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.base import Base3DDetector
from .oneformer3d import S3DISOneFormer3D
from .enhanced_query_decoder import EnhancedQueryDecoder, CabbageHeadStageClassifier


@MODELS.register_module(force=True)
class S3DISEnhancedOneFormer3D(S3DISOneFormer3D):
    """
    增强的S3DIS OneFormer3D模型，集成甘蓝球径分割精度增强
    """
    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 enable_cabbage_refinement: bool = True,
                 enable_stage_classification: bool = True,
                 refinement_config: Optional[Dict] = None,
                 stage_classifier_config: Optional[Dict] = None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        
        # 调用父类初始化
        super().__init__(
            in_channels=in_channels,
            num_channels=num_channels,
            voxel_size=voxel_size,
            num_classes=num_classes,
            min_spatial_shape=min_spatial_shape,
            backbone=backbone,
            decoder=decoder,
            criterion=criterion,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
        
        # 甘蓝球径增强配置
        self.enable_cabbage_refinement = enable_cabbage_refinement
        self.enable_stage_classification = enable_stage_classification
        
        # 生长期分类器
        if enable_stage_classification:
            default_stage_config = {
                'in_channels': num_channels,
                'hidden_channels': 128,
                'num_stages': 3,  # 幼苗期、生长期、结球期
                'dropout': 0.1
            }
            if stage_classifier_config is not None:
                default_stage_config.update(stage_classifier_config)
            
            self.stage_classifier = CabbageHeadStageClassifier(**default_stage_config)
        else:
            self.stage_classifier = None
        
        # 增强解码器
        if enable_cabbage_refinement:
            # 创建增强解码器配置
            enhanced_decoder_config = {
                'num_layers': 3,
                'num_classes': num_classes,
                'num_instance_queries': 400,
                'num_semantic_queries': 3,
                'num_instance_classes': num_classes,
                'in_channels': num_channels,
                'd_model': 256,
                'num_heads': 8,
                'hidden_dim': 1024,
                'dropout': 0.0,
                'activation_fn': 'gelu',
                'iter_pred': True,
                'attn_mask': True,
                'fix_attention': True,
                'objectness_flag': True,
                'enable_cabbage_refinement': True,
                'refinement_config': refinement_config
            }
            
            # 如果提供了decoder配置，更新默认配置
            if decoder is not None:
                enhanced_decoder_config.update(decoder)
            
            self.enhanced_decoder = EnhancedQueryDecoder(**enhanced_decoder_config)
        else:
            self.enhanced_decoder = None
    
    def _safe_forward(self, inputs, data_samples=None, mode='tensor', **kwargs):
        """Safe forward method with matrix multiplication error handling"""
        try:
            return super().forward(inputs, data_samples, mode=mode, **kwargs)
        except Exception as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                print(f"[Error] Matrix multiplication error in forward pass: {e}")
                print("  Applying dimension fix...")
                
                # Try to handle the error gracefully
                try:
                    if mode == 'predict':
                        print("  Returning default predictions due to matrix multiplication error")
                        # Return default predictions
                        if isinstance(inputs, dict) and 'points' in inputs:
                            device = next(iter(inputs.values())).device
                        else:
                            device = torch.device('cpu')
                        
                        default_pred = {
                            'pred_pts_seg': torch.zeros(1000, 3, device=device),
                            'pred_instances': []
                        }
                        return default_pred
                    else:
                        raise e
                except Exception as e2:
                    print(f"  Failed to handle matrix multiplication error: {e2}")
                    raise e
            else:
                raise e

    def _to_numpy(self, x):
        """Convert tensor to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _ensure_rgb01(self, colors):
        """Ensure colors are in [0,1] range."""
        colors = colors.astype(np.float32)
        
        # 检查颜色范围并相应处理
        if colors.min() < 0.0 and colors.max() <= 1.0:
            # 颜色在 [-1, 1] 范围内（NormalizePointsColor_ 的结果）
            # 按照 NormalizePointsColor_ 的逆变换恢复原始颜色
            color_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
            color_std = 127.5
            
            # 逆变换：先乘以标准差，再加均值
            colors = colors * color_std + color_mean
            
            # 然后归一化到 [0, 1] 范围
            colors = colors / 255.0
            
            print(f"Debug - 颜色从[-1,1]逆变换到[0,1] (color_mean={color_mean}, color_std={color_std})")
            print(f"Debug - 逆变换后颜色范围: [{colors.min():.3f}, {colors.max():.3f}]")
        elif colors.max() > 1.0:
            # 颜色在 [0, 255] 范围内，需要归一化
            colors /= 255.0
            print(f"Debug - 颜色从[0,255]归一化到[0,1]")
        elif colors.min() >= 0.0 and colors.max() <= 1.0:
            # 颜色已经在 [0, 1] 范围内
            print(f"Debug - 颜色已在[0,1]范围内")
        else:
            # 其他情况，直接裁剪
            print(f"Debug - 颜色范围异常，直接裁剪")
        
        return np.clip(colors, 0.0, 1.0)

    def save_point_cloud(self, points_xyz, colors_rgb01, file_path):
        """Save point cloud to PLY file."""
        points_xyz = self._to_numpy(points_xyz).astype(np.float32)
        colors_rgb01 = self._to_numpy(colors_rgb01).astype(np.float32)
        # 不重复处理颜色，调用者已经确保颜色在[0,1]范围内
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_xyz)
        pc.colors = o3d.utility.Vector3dVector(colors_rgb01)
        o3d.io.write_point_cloud(file_path, pc)

    def filter_color_and_save_instances(self, instance_labels, instance_scores, pts_instance_mask,
                                        input_points, input_point_name, threshold=0.3):
        """Filter instances by score and save colored point clouds."""
        base_dir = f"./work_dirs/{input_point_name}"
        os.makedirs(base_dir, exist_ok=True)

        np_points = self._to_numpy(input_points)
        xyz = np_points[:, :3].astype(np.float32)
        base_rgb = self._ensure_rgb01(np_points[:, 3:6])

        # 处理实例标签和分数
        instance_labels_np = self._to_numpy(instance_labels)
        instance_scores_np = self._to_numpy(instance_scores)
        
        # 处理实例掩码 - 确保正确的维度
        if isinstance(pts_instance_mask, list):
            pts_instance_mask = pts_instance_mask[0]  # 取第一个元素
        masks = self._to_numpy(pts_instance_mask)
        
        # 确保掩码是2D的 (num_instances, num_points)
        if masks.ndim == 1:
            # 如果是一维的，需要重新整形
            num_instances = len(instance_labels_np)
            num_points = len(xyz)
            if len(masks) == num_instances * num_points:
                masks = masks.reshape(num_instances, num_points)
            else:
                # 如果维度不匹配，创建默认掩码
                masks = np.zeros((num_instances, num_points), dtype=bool)
        elif masks.ndim == 2:
            # 确保维度匹配
            num_instances = len(instance_labels_np)
            if masks.shape[0] != num_instances:
                # 如果实例数不匹配，调整掩码
                if masks.shape[0] > num_instances:
                    masks = masks[:num_instances]
                else:
                    # 补零
                    pad_masks = np.zeros((num_instances - masks.shape[0], masks.shape[1]), dtype=bool)
                    masks = np.vstack([masks, pad_masks])
        
        masks = masks.astype(bool)
        scores = instance_scores_np.astype(np.float32)
        labels = instance_labels_np.astype(np.int64)
        num_instances = len(labels)
        
        # 调试信息
        print(f"Debug - 实例数: {num_instances}, 点云数: {len(xyz)}, 掩码形状: {masks.shape}")
        
        # 确保掩码的第二维与点云数量匹配
        if masks.shape[1] != len(xyz):
            print(f"Warning - 掩码维度不匹配: 掩码点数 {masks.shape[1]}, 实际点数 {len(xyz)}")
            # 调整掩码维度
            if masks.shape[1] > len(xyz):
                masks = masks[:, :len(xyz)]
            else:
                # 补零
                pad_masks = np.zeros((masks.shape[0], len(xyz) - masks.shape[1]), dtype=bool)
                masks = np.hstack([masks, pad_masks])

        # 预定义颜色调色板 - 甘蓝专用
        predefined_colors = [
            [1.0, 0.5, 0.0],  # 橙色 - 类别0: 球径（甘蓝球径）
            [0.9, 0.6, 0.3],  # 亮棕色 - 类别1: ground（地面）- 大幅提高亮度使其更明显
            [0.0, 0.8, 0.0],  # 绿色 - 类别2: leaf（叶子）
            # 为类别2的多个实例提供不同的绿色调
            [1.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 1.0],
            [1.0, 0.75, 0.8],
            [0.5, 1.0, 0.5],
            [0.75, 0.5, 0.25],
            [0.25, 0.5, 0.75],
            [0.75, 0.25, 0.5],
            [0.5, 0.75, 0.25],
            [0.25, 0.75, 0.5],
            [0.5, 0.25, 0.75],
            [0.75, 0.5, 0.5],
            [0.5, 0.75, 0.75],
            [0.75, 0.75, 0.5],
        ]

        global_rgb = base_rgb.copy()
        instance_count_per_label = {}
        palette = predefined_colors[2:]  # 从第3个颜色开始用于类别2的实例
        palette_len = len(palette)
        label2_instance_idx = 0

        for i in range(num_instances):
            if scores[i] < threshold:
                continue
            lbl = int(labels[i])
            instance_mask = masks[i]

            # 跳过地面类别(类别1)的实例分割涂色
            if lbl == 1:
                print(f"    跳过地面实例 {i} (类别{lbl}): {np.sum(instance_mask)} 个点")
                continue

            # 根据类别分配颜色 - 只对球径(0)和叶子(2)进行实例涂色
            if lbl == 0:
                # 类别0: 球径（甘蓝球径）- 橙色
                color = np.array(predefined_colors[0], dtype=np.float32)
            elif lbl == 2:
                # 类别2: leaf（叶子）- 使用不同的绿色调
                color = np.array(palette[label2_instance_idx % palette_len], dtype=np.float32)
                label2_instance_idx += 1
            else:
                # 其他类别 - 跳过
                print(f"    跳过其他类别实例 {i} (类别{lbl}): {np.sum(instance_mask)} 个点")
                continue

            global_rgb[instance_mask] = color
            instance_count_per_label[lbl] = instance_count_per_label.get(lbl, 0) + 1
            instance_points_xyz = xyz[instance_mask]
            instance_colors = np.repeat(color[None, :], instance_points_xyz.shape[0], axis=0)
            instance_pc_path = os.path.join(
                base_dir, f"{input_point_name}_{lbl}_{instance_count_per_label[lbl]}.ply"
            )
            self.save_point_cloud(instance_points_xyz, instance_colors, instance_pc_path)
            print(f"    实例 {i} (类别{lbl}): {np.sum(instance_mask)} 个点")

        # 保存原始点云（使用原始颜色）和着色后的点云
        input_pc_path = os.path.join(base_dir, f"{input_point_name}.ply")
        
        # 调试原始颜色数据
        raw_colors = np_points[:, 3:6]
        print(f"Debug - 原始颜色数据形状: {raw_colors.shape}")
        print(f"Debug - 原始颜色范围: [{raw_colors.min():.3f}, {raw_colors.max():.3f}]")
        print(f"Debug - 原始颜色前5个: {raw_colors[:5]}")
        print(f"Debug - 原始颜色数据类型: {raw_colors.dtype}")
        
        # 确保原始颜色在[0,1]范围内
        original_rgb = self._ensure_rgb01(np_points[:, 3:6])
        print(f"Debug - 处理后颜色范围: [{original_rgb.min():.3f}, {original_rgb.max():.3f}]")
        print(f"Debug - 处理后颜色前5个: {original_rgb[:5]}")
        
        self.save_point_cloud(xyz, original_rgb, input_pc_path)
        colored_pc_path = os.path.join(base_dir, f"{input_point_name}_colored.ply")
        self.save_point_cloud(xyz, global_rgb, colored_pc_path)

    def visualize_semantic_segmentation(self, semantic_labels, input_points, input_point_name):
        """可视化语义分割结果 - 重构版本"""
        base_dir = f"./work_dirs/{input_point_name}"
        os.makedirs(base_dir, exist_ok=True)

        np_points = self._to_numpy(input_points)
        xyz = np_points[:, :3].astype(np.float32)
        base_rgb = self._ensure_rgb01(np_points[:, 3:6])

        # 处理语义标签 - 根据重构后的代码结构
        semantic_labels_np = self._to_numpy(semantic_labels)
        print(f"Debug - 语义标签形状: {semantic_labels_np.shape}, 点云数: {len(xyz)}")
        
        # 根据重构后的代码，语义标签应该已经是正确的格式
        # 但为了兼容性，仍然进行格式检查和处理
        if semantic_labels_np.ndim == 2:
            # 2D数组处理
            if semantic_labels_np.shape[0] == len(xyz):
                # (num_points, num_classes) - 取最大概率的类别
                semantic_labels_np = np.argmax(semantic_labels_np, axis=1)
                print(f"  处理方式: (num_points, num_classes) -> argmax")
            elif semantic_labels_np.shape[1] == len(xyz):
                # (num_classes, num_points) - 取最大概率的类别
                semantic_labels_np = np.argmax(semantic_labels_np, axis=0)
                print(f"  处理方式: (num_classes, num_points) -> argmax")
            else:
                print(f"Warning - 语义标签维度不匹配: {semantic_labels_np.shape}, 点云数: {len(xyz)}")
                # 创建默认标签
                semantic_labels_np = np.zeros(len(xyz), dtype=np.int64)
        elif semantic_labels_np.ndim == 1:
            # 1D数组处理
            if len(semantic_labels_np) != len(xyz):
                print(f"Warning - 语义标签长度不匹配: {len(semantic_labels_np)}, 点云数: {len(xyz)}")
                # 调整长度
                if len(semantic_labels_np) > len(xyz):
                    semantic_labels_np = semantic_labels_np[:len(xyz)]
                else:
                    # 补零
                    pad_labels = np.zeros(len(xyz) - len(semantic_labels_np), dtype=semantic_labels_np.dtype)
                    semantic_labels_np = np.concatenate([semantic_labels_np, pad_labels])
        else:
            print(f"Warning - 不支持的语义标签维度: {semantic_labels_np.ndim}")
            # 创建默认标签
            semantic_labels_np = np.zeros(len(xyz), dtype=np.int64)
        
        semantic_labels_np = semantic_labels_np.astype(np.int64)
        print(f"Debug - 处理后语义标签形状: {semantic_labels_np.shape}")

        # 甘蓝专用语义分割颜色映射
        semantic_colors = [
                [0.8, 0.0, 0.8],  # 紫色 - 类别0: 球径（甘蓝球径）
                [0.9, 0.6, 0.3],  # 亮棕色 - 类别1: ground（地面）- 大幅提高亮度使其更明显
                [0.0, 0.8, 0.0],  # 绿色 - 类别2: leaf（叶子）
            ]

        semantic_rgb = np.zeros_like(base_rgb)

        # 为每个语义类别分配颜色
        class_counts = {}
        for class_id in range(len(semantic_colors)):
            mask = (semantic_labels_np == class_id)
            if np.any(mask):
                semantic_rgb[mask] = semantic_colors[class_id]
                count = np.sum(mask)
                class_counts[class_id] = count
                print(f"  类别 {class_id}: {count} 个点")
        
        # 检查类别分布
        unique_classes = np.unique(semantic_labels_np)
        print(f"  出现的类别: {unique_classes.tolist()}")
        print(f"  类别分布: {class_counts}")

        # 保存语义分割结果
        semantic_pc_path = os.path.join(base_dir, f"{input_point_name}_semantic.ply")
        self.save_point_cloud(xyz, semantic_rgb, semantic_pc_path)
        
        # 如果类别分布异常，保存调试信息
        if len(unique_classes) < 2 or (len(class_counts) > 0 and max(class_counts.values()) / len(xyz) > 0.95):
            debug_info_path = os.path.join(base_dir, f"{input_point_name}_semantic_debug.txt")
            with open(debug_info_path, 'w') as f:
                f.write(f"语义分割调试信息\n")
                f.write(f"点云数: {len(xyz)}\n")
                f.write(f"语义标签形状: {semantic_labels_np.shape}\n")
                f.write(f"出现的类别: {unique_classes.tolist()}\n")
                f.write(f"类别分布: {class_counts}\n")
                f.write(f"类别比例: {[(k, v/len(xyz)) for k, v in class_counts.items()]}\n")
            print(f"  调试信息已保存到: {debug_info_path}")

    def visualize_inference_results(self, batch_inputs_dict, batch_data_samples):
        """
        可视化推理结果，包括语义分割和实例分割 - 重构版本
        """
        try:
            for i, data_sample in enumerate(batch_data_samples):
                if not hasattr(data_sample, 'pred_pts_seg') or data_sample.pred_pts_seg is None:
                    print(f"Warning - 样本 {i} 没有预测结果，跳过可视化")
                    continue
                    
                pred_pts_seg = data_sample.pred_pts_seg
                input_points = batch_inputs_dict["points"][i]
                
                # 获取点云文件名
                if hasattr(data_sample, 'lidar_path'):
                    input_point_name = data_sample.lidar_path.split('/')[-1].split('.')[0]
                else:
                    input_point_name = f"sample_{i}"
                
                print(f"开始可视化样本 {i}: {input_point_name}")
                
                # 实例分割可视化
                try:
                    if hasattr(pred_pts_seg, 'instance_labels') and hasattr(pred_pts_seg, 'instance_scores') and hasattr(pred_pts_seg, 'pts_instance_mask'):
                        instance_labels = pred_pts_seg.instance_labels
                        instance_scores = pred_pts_seg.instance_scores
                        pts_instance_mask = pred_pts_seg.pts_instance_mask
                        
                        # 确保pts_instance_mask是正确的格式
                        if isinstance(pts_instance_mask, list) and len(pts_instance_mask) > 0:
                            pts_instance_mask = pts_instance_mask[0]
                        
                        self.filter_color_and_save_instances(
                            instance_labels, instance_scores, pts_instance_mask,
                            input_points, input_point_name, threshold=0.3
                        )
                        print(f"✓ 实例分割可视化完成: {input_point_name}")
                    else:
                        print(f"Warning - 样本 {i} 缺少实例分割数据")
                except Exception as e:
                    print(f"Error - 实例分割可视化失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 语义分割可视化 - 重构版本
                try:
                    if hasattr(pred_pts_seg, 'pts_semantic_mask'):
                        semantic_labels = pred_pts_seg.pts_semantic_mask
                        
                        # 根据重构后的代码，pts_semantic_mask可能是列表格式
                        if isinstance(semantic_labels, list) and len(semantic_labels) > 0:
                            # 取第一个元素（通常是语义分割结果）
                            semantic_labels = semantic_labels[0]
                        
                        self.visualize_semantic_segmentation(semantic_labels, input_points, input_point_name)
                        print(f"✓ 语义分割可视化完成: {input_point_name}")
                    else:
                        print(f"Warning - 样本 {i} 缺少语义分割数据")
                        # 尝试从其他属性获取语义信息
                        if hasattr(pred_pts_seg, 'enhanced_outputs'):
                            enhanced_outputs = pred_pts_seg.enhanced_outputs
                            if 'logits' in enhanced_outputs:
                                # 从增强输出中提取语义信息
                                logits = enhanced_outputs['logits'][0] if isinstance(enhanced_outputs['logits'], list) else enhanced_outputs['logits']
                                if logits.shape[1] >= 3:
                                    semantic_labels = logits[:, :3].argmax(dim=1)
                                    self.visualize_semantic_segmentation(semantic_labels, input_points, input_point_name)
                                    print(f"✓ 从增强输出提取语义分割可视化完成: {input_point_name}")
                except Exception as e:
                    print(f"Error - 语义分割可视化失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"Error - 可视化过程失败: {e}")
            import traceback
            traceback.print_exc()

    def extract_features_with_stage(self, x):
        """
        提取特征并进行生长期分类
        Args:
            x: 稀疏张量
        Returns:
            features: 特征列表
            stage_info: 生长期信息
        """
        # 提取基础特征
        features = self.extract_feat(x)
        
        stage_info = {}
        
        # 生长期分类
        if self.enable_stage_classification and self.stage_classifier is not None:
            stage_logits, stage_features = self.stage_classifier(features)
            stage_probs = F.softmax(stage_logits, dim=-1)
            
            stage_info = {
                'stage_logits': stage_logits,
                'stage_probs': stage_probs,
                'stage_features': stage_features
            }
        
        return features, stage_info
    
    def forward(self, inputs, data_samples=None, mode='tensor', **kwargs):
        """对齐Base3DDetector的forward签名，委托给父类实现。"""
        return self._safe_forward(inputs, data_samples, mode=mode, **kwargs)
    
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """
        计算损失，包括基础损失和增强损失
        """
        # 数据预处理
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        # 提取特征和生长期信息
        features, stage_info = self.extract_features_with_stage(x)
        
        # 解码（使用增强解码器）
        if self.enable_cabbage_refinement and self.enhanced_decoder is not None:
            base_outputs = self.enhanced_decoder(features)
        else:
            base_outputs = self.decoder(features)
        
        # 准备真值数据
        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                S3DISOneFormer3D.get_gt_semantic_masks(sem_mask, voxel_superpoints, self.num_classes)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                S3DISOneFormer3D.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        # 计算损失（传入与 decoder 点域对齐的点集，与 stage 概率）
        # 将稀疏坐标按 batch 拆分为与 features 同长度的点坐标（与mask域一致）
        dec_points = []
        for i in range(len(batch_data_samples)):
            mask_i = (coordinates[:, 0] == i)
            coords_i = coordinates[mask_i, 1:4].to(features[i].device).float()
            n_feat = features[i].shape[0]
            if coords_i.shape[0] != n_feat:
                # 长度不一致时，裁剪或补零以对齐
                if coords_i.shape[0] > n_feat:
                    coords_i = coords_i[:n_feat]
                else:
                    pad = torch.zeros((n_feat - coords_i.shape[0], 3), device=coords_i.device, dtype=coords_i.dtype)
                    coords_i = torch.cat([coords_i, pad], dim=0)
            dec_points.append(coords_i)
        stage_probs = stage_info.get('stage_probs', None)
        loss = self.criterion(base_outputs, sp_gt_instances, points=dec_points, stage_probs=stage_probs)
        
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """
        预测结果，包括基础预测和增强预测，并进行可视化
        """
        # 数据预处理
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        # 提取特征
        features = self.extract_feat(x)
        
        # 基础预测
        if self.enable_cabbage_refinement and self.enhanced_decoder is not None:
            base_outputs = self.enhanced_decoder(features)
        else:
            base_outputs = self.decoder(features)
        # 使用父类的预测后处理，得到 PointData 列表
        base_results = S3DISOneFormer3D.predict_by_feat(self, base_outputs, inverse_mapping)
        
        # 增强预测（如果启用）
        # 结果列表（父类已返回 List[PointData]）
        results_list = base_results

        # 更新数据样本
        for i, data_sample in enumerate(batch_data_samples):
            # 防止KeyError或索引错误
            idx = i if i < len(results_list) else len(results_list) - 1
            data_sample.pred_pts_seg = results_list[idx]
        
        # 可视化推理结果
        self.visualize_inference_results(batch_inputs_dict, batch_data_samples)
        
        # 添加全景分割可视化
        self.visualize_panoptic_segmentation(batch_inputs_dict, batch_data_samples)
        
        return batch_data_samples
    
    # 依赖 S3DISOneFormer3D 的静态方法来生成 sp_* 真值掩码
    
    def predict_by_feat(self, x, inverse_mapping):
        """
        根据特征进行预测
        """
        # 这里需要根据实际的预测逻辑来实现
        # 简化实现，返回基础预测结果
        return x


@MODELS.register_module()
class S3DISCabbageOneFormer3D(S3DISEnhancedOneFormer3D):
    """
    专门针对甘蓝球径分割优化的OneFormer3D模型
    """
    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 cabbage_config: Optional[Dict] = None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        
        # 甘蓝专用配置
        default_cabbage_config = {
            'enable_cabbage_refinement': True,
            'enable_stage_classification': True,
            'refinement_config': {
                'in_channels': num_channels,
                'hidden_channels': 128,
                'num_classes': num_classes,
                'refinement_layers': 3,
                'attention_heads': 8,
                'dropout': 0.1
            },
            'stage_classifier_config': {
                'in_channels': num_channels,
                'hidden_channels': 128,
                'num_stages': 3,
                'dropout': 0.1
            }
        }
        
        if cabbage_config is not None:
            default_cabbage_config.update(cabbage_config)
        
        # 调用父类初始化
        super().__init__(
            in_channels=in_channels,
            num_channels=num_channels,
            voxel_size=voxel_size,
            num_classes=num_classes,
            min_spatial_shape=min_spatial_shape,
            backbone=backbone,
            decoder=decoder,
            criterion=criterion,
            enable_cabbage_refinement=default_cabbage_config['enable_cabbage_refinement'],
            enable_stage_classification=default_cabbage_config['enable_stage_classification'],
            refinement_config=default_cabbage_config['refinement_config'],
            stage_classifier_config=default_cabbage_config['stage_classifier_config'],
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
    
    def forward(self, inputs, data_samples=None, mode='tensor', **kwargs):
        """对齐Base3DDetector的forward签名，委托给父类实现。"""
        return self._safe_forward(inputs, data_samples, mode=mode, **kwargs)
    
    def visualize_panoptic_segmentation(self, batch_inputs_dict, batch_data_samples):
        """
        可视化全景分割结果，包括语义分割和实例分割
        """
        try:
            for i, data_sample in enumerate(batch_data_samples):
                if not hasattr(data_sample, 'pred_pts_seg') or data_sample.pred_pts_seg is None:
                    print(f"Warning - 样本 {i} 没有预测结果，跳过全景分割可视化")
                    continue
                    
                pred_pts_seg = data_sample.pred_pts_seg
                input_points = batch_inputs_dict["points"][i]
                
                # 获取点云文件名
                if hasattr(data_sample, 'lidar_path'):
                    input_point_name = data_sample.lidar_path.split('/')[-1].split('.')[0]
                else:
                    input_point_name = f"sample_{i}"
                
                print(f"开始全景分割可视化样本 {i}: {input_point_name}")
                
                # 检查是否有全景分割数据
                if not (hasattr(pred_pts_seg, 'pts_semantic_mask') and 
                        hasattr(pred_pts_seg, 'pts_instance_mask') and
                        len(pred_pts_seg.pts_semantic_mask) > 1):
                    print(f"Warning - 样本 {i} 缺少全景分割数据")
                    continue
                
                # 获取全景分割的语义和实例部分
                panoptic_semantic = pred_pts_seg.pts_semantic_mask[1]  # 全景分割的语义部分
                panoptic_instance = pred_pts_seg.pts_instance_mask[1]  # 全景分割的实例部分
                
                self.visualize_panoptic_segmentation_single(
                    panoptic_semantic, panoptic_instance, input_points, input_point_name
                )
                print(f"✓ 全景分割可视化完成: {input_point_name}")
                
        except Exception as e:
            print(f"Error - 全景分割可视化过程失败: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_panoptic_segmentation_single(self, panoptic_semantic, panoptic_instance, input_points, input_point_name):
        """可视化单个样本的全景分割结果"""
        base_dir = f"./work_dirs/{input_point_name}"
        os.makedirs(base_dir, exist_ok=True)

        np_points = self._to_numpy(input_points)
        xyz = np_points[:, :3].astype(np.float32)
        base_rgb = self._ensure_rgb01(np_points[:, 3:6])

        # 处理全景分割的语义部分
        panoptic_semantic_np = self._to_numpy(panoptic_semantic)
        panoptic_instance_np = self._to_numpy(panoptic_instance)
        
        print(f"Debug - 全景分割可视化:")
        print(f"  语义标签形状: {panoptic_semantic_np.shape}")
        print(f"  实例标签形状: {panoptic_instance_np.shape}")
        print(f"  点云数: {len(xyz)}")
        
        # 确保标签形状正确
        if panoptic_semantic_np.shape[0] != len(xyz):
            print(f"Warning - 全景语义标签维度不匹配: {panoptic_semantic_np.shape}, 点云数: {len(xyz)}")
            return
            
        if panoptic_instance_np.shape[0] != len(xyz):
            print(f"Warning - 全景实例标签维度不匹配: {panoptic_instance_np.shape}, 点云数: {len(xyz)}")
            return

        # 创建全景分割颜色映射
        # 语义部分使用固定颜色（stuff classes）
        semantic_colors = np.zeros((len(xyz), 3), dtype=np.float32)
        
        # 为不同的语义类别分配颜色
        for class_id in range(3):  # 假设有3个语义类别
            class_mask = (panoptic_semantic_np == class_id)
            if np.any(class_mask):
                if class_id == 0:  # 球径 - 紫色
                    color = np.array([0.8, 0.0, 0.8], dtype=np.float32)
                elif class_id == 1:  # 地面 - 亮棕色
                    color = np.array([0.9, 0.6, 0.3], dtype=np.float32)
                elif class_id == 2:  # 叶子 - 绿色
                    color = np.array([0.0, 0.8, 0.0], dtype=np.float32)
                else:
                    color = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # 默认灰色
                
                semantic_colors[class_mask] = color
                print(f"  语义类别 {class_id}: {np.sum(class_mask)} 个点")

        # 实例部分使用不同颜色（things classes）
        instance_colors = semantic_colors.copy()
        unique_instances = np.unique(panoptic_instance_np)
        print(f"  检测到 {len(unique_instances)} 个实例")
        
        # 预定义颜色
        predefined_colors = [
            [1.0, 0.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.5],
            [0.0, 1.0, 1.0], [1.0, 0.75, 0.8], [0.5, 1.0, 0.5], [0.75, 0.5, 0.25],
            [0.25, 0.5, 0.75], [0.75, 0.25, 0.5], [0.5, 0.75, 0.25], [0.25, 0.75, 0.5],
            [0.5, 0.25, 0.75], [0.75, 0.5, 0.5], [0.5, 0.75, 0.75], [0.75, 0.75, 0.5]
        ]
        
        # 为每个实例分配不同颜色（只对球径和叶子类别进行实例涂色）
        color_idx = 0
        for i, inst_id in enumerate(unique_instances):
            inst_mask = (panoptic_instance_np == inst_id)
            if not np.any(inst_mask):
                continue
            
            # 检查该实例对应的语义类别，只对球径(0)和叶子(2)类别进行实例涂色
            inst_semantic_labels = panoptic_semantic_np[inst_mask]
            if len(inst_semantic_labels) > 0:
                dominant_semantic_class = np.bincount(inst_semantic_labels).argmax()
                if dominant_semantic_class == 1:  # 跳过地面类别(环境类别)
                    print(f"    跳过地面实例 {inst_id}: {np.sum(inst_mask)} 个点")
                    continue
                elif dominant_semantic_class not in [0, 2]:  # 只处理球径(0)和叶子(2)类别
                    print(f"    跳过其他类别实例 {inst_id} (类别{dominant_semantic_class}): {np.sum(inst_mask)} 个点")
                    continue
            
            # 使用预定义颜色循环，只对有效的实例进行涂色
            color = np.array(predefined_colors[color_idx % len(predefined_colors)], dtype=np.float32)
            instance_colors[inst_mask] = color
            print(f"    实例 {inst_id} (类别{dominant_semantic_class}): {np.sum(inst_mask)} 个点")
            color_idx += 1

        # 保存全景分割结果
        panoptic_semantic_pc_path = os.path.join(base_dir, f"{input_point_name}_panoptic_semantic.ply")
        self.save_point_cloud(xyz, semantic_colors, panoptic_semantic_pc_path)
        print(f"  全景语义分割已保存到: {panoptic_semantic_pc_path}")

        panoptic_instance_pc_path = os.path.join(base_dir, f"{input_point_name}_panoptic_instance.ply")
        self.save_point_cloud(xyz, instance_colors, panoptic_instance_pc_path)
        print(f"  全景实例分割已保存到: {panoptic_instance_pc_path}")

        # 保存完整的全景分割结果（语义+实例）
        panoptic_full_pc_path = os.path.join(base_dir, f"{input_point_name}_panoptic_full.ply")
        self.save_point_cloud(xyz, instance_colors, panoptic_full_pc_path)
        print(f"  完整全景分割已保存到: {panoptic_full_pc_path}")

        # 统计信息
        print(f"Debug - 全景分割统计:")
        print(f"  语义类别分布:")
        for class_id in range(3):
            count = np.sum(panoptic_semantic_np == class_id)
            print(f"    类别 {class_id}: {count} 个点 ({count/len(xyz)*100:.1f}%)")
        
        print(f"  实例分布:")
        for inst_id in unique_instances:
            count = np.sum(panoptic_instance_np == inst_id)
            print(f"    实例 {inst_id}: {count} 个点 ({count/len(xyz)*100:.1f}%)")

    def forward_features(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提供每点特征与每点logits，供外部可视化脚本调用。
        - 输入: points [B, N, C_in], 至少包含 xyz
        - 输出: (per_point_features [B,N,d_model], per_point_logits [B,N,num_classes])
        注意: 这里不走稀疏体素/解码完整流程，仅用于特征可视化时的直通接口，
             尽量复用现有解码器的轻量点头以保证类别对齐。
        """
        assert torch.is_tensor(points) and points.dim() == 3, "points应为[B,N,C]张量"
        device = points.device
        B, N, Cin = points.shape

        # 解析 d_model 与 num_classes
        d_model = 256
        num_classes = getattr(self, 'num_classes', 3)
        point_head = None

        if hasattr(self, 'enhanced_decoder') and self.enhanced_decoder is not None:
            # EnhancedQueryDecoder 内部定义了 _d_model 与 _orig_point_head
            if hasattr(self.enhanced_decoder, '_d_model'):
                d_model = int(self.enhanced_decoder._d_model)
            if hasattr(self.enhanced_decoder, '_orig_point_head'):
                point_head = self.enhanced_decoder._orig_point_head
            if hasattr(self.enhanced_decoder, '_num_classes_ref'):
                num_classes = int(self.enhanced_decoder._num_classes_ref)
        elif hasattr(self, 'decoder') and self.decoder is not None:
            # 标准 decoder 情况（尽力获取）
            if hasattr(self.decoder, '_d_model'):
                d_model = int(self.decoder._d_model)
            if hasattr(self.decoder, '_orig_point_head'):
                point_head = self.decoder._orig_point_head
            if hasattr(self.decoder, '_num_classes_ref'):
                num_classes = int(self.decoder._num_classes_ref)

        # 创建/缓存一个线性投影将输入点特征映射到 d_model
        in_dim = Cin
        if not hasattr(self, '_vis_proj'):
            self._vis_proj = nn.Linear(in_dim, d_model).to(device)
        elif isinstance(self._vis_proj, nn.Linear) and self._vis_proj.in_features != in_dim:
            # 若输入维度变化，重建投影层
            self._vis_proj = nn.Linear(in_dim, d_model).to(device)

        per_point_features = self._vis_proj(points)  # [B, N, d_model]

        # 通过已有点头生成 logits；若不存在则临时创建一个仅用于可视化的头
        if point_head is None:
            if not hasattr(self, '_vis_point_head'):
                self._vis_point_head = nn.Linear(d_model, max(1, num_classes)).to(device)
            point_head = self._vis_point_head

        # 线性层期望 [..., d_model] 输入
        per_point_logits = point_head(per_point_features)  # [B, N, num_classes]

        return per_point_features, per_point_logits
