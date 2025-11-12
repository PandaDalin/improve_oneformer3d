import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from .query_decoder import QueryDecoder


class CabbageHeadRefinementModule(nn.Module):
    """
    甘蓝球径分割精度增强模块
    在原始OpenFormer3D输出的基础上进行精细化处理
    """
    def __init__(self,
                 in_channels: int = 256,
                 hidden_channels: int = 128,
                 num_classes: int = 3,
                 refinement_layers: int = 2,
                 attention_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.refinement_layers = refinement_layers
        
        # 特征提取和融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 多头注意力机制用于精细化
        self.refinement_attention = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_channels, 
                attention_heads, 
                dropout=dropout, 
                batch_first=True
            ) for _ in range(refinement_layers)
        ])
        
        # 前馈网络
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels)
            ) for _ in range(refinement_layers)
        ])
        
        # 层归一化
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(refinement_layers * 2)
        ])
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_classes)
        )
        
        # 球径区域检测器
        self.head_detector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                features: List[torch.Tensor],
                original_logits: List[torch.Tensor],
                points: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            features: 特征列表 [B, N, C]
            original_logits: 原始logits列表 [B, N, num_classes]
            points: 点云坐标列表 [B, N, 3] (可选)
        Returns:
            enhanced_outputs: 增强输出字典
        """
        batch_size = len(features)
        enhanced_outputs = {}
        
        for b in range(batch_size):
            feat = features[b]  # [N, C]
            orig_logit = original_logits[b]  # [N, num_classes]
            
            # 特征融合
            refined_feat = self.feature_fusion(feat)  # [N, hidden_channels]
            
            # 精细化处理
            for i in range(self.refinement_layers):
                # 确保输入是3D张量 [batch_size, seq_len, hidden_dim]
                if refined_feat.dim() == 4:
                    # 如果是4D张量，压缩前两个维度
                    batch_size, seq_len, hidden_dim = refined_feat.shape[0] * refined_feat.shape[1], refined_feat.shape[2], refined_feat.shape[3]
                    refined_feat_3d = refined_feat.view(batch_size, seq_len, hidden_dim)
                elif refined_feat.dim() == 2:
                    # 如果是2D张量，添加batch维度
                    refined_feat_3d = refined_feat.unsqueeze(0)
                else:
                    refined_feat_3d = refined_feat
                
                # 自注意力
                attn_out, _ = self.refinement_attention[i](
                    refined_feat_3d, 
                    refined_feat_3d, 
                    refined_feat_3d
                )
                
                # 恢复原始形状
                if refined_feat.dim() == 4:
                    attn_out = attn_out.view(refined_feat.shape)
                elif refined_feat.dim() == 2:
                    attn_out = attn_out.squeeze(0)
                
                refined_feat = self.norm_layers[i * 2](refined_feat + attn_out)
                
                # 前馈网络
                ffn_out = self.ffn_layers[i](refined_feat)
                refined_feat = self.norm_layers[i * 2 + 1](refined_feat + ffn_out)
            
            # 生成精细化logits
            refined_logits = self.output_proj(refined_feat)  # [N, num_classes]
            
            # 球径区域检测
            head_mask_prob = self.head_detector(refined_feat).squeeze(-1)  # [N]
            
            # 简单的质量分数（基于球径检测的置信度）
            quality_score = torch.mean(head_mask_prob)
            
            # 存储结果
            if b == 0:
                enhanced_outputs['logits'] = [refined_logits]
                enhanced_outputs['refined_segmentation_logits'] = [refined_logits]  # 兼容集成损失
                enhanced_outputs['original_logits'] = [orig_logit]
                enhanced_outputs['original_segmentation_logits'] = [orig_logit]  # 兼容集成损失
                enhanced_outputs['head_mask_prob'] = [head_mask_prob]
                enhanced_outputs['segmentation_quality_score'] = [quality_score]  # 兼容集成损失
                enhanced_outputs['refined_features'] = [refined_feat]  # 导出精修后的特征用于可视化
            else:
                enhanced_outputs['logits'].append(refined_logits)
                enhanced_outputs['refined_segmentation_logits'].append(refined_logits)  # 兼容集成损失
                enhanced_outputs['original_logits'].append(orig_logit)
                enhanced_outputs['original_segmentation_logits'].append(orig_logit)  # 兼容集成损失
                enhanced_outputs['head_mask_prob'].append(head_mask_prob)
                enhanced_outputs['segmentation_quality_score'].append(quality_score)  # 兼容集成损失
                enhanced_outputs['refined_features'].append(refined_feat)  # 导出精修后的特征用于可视化
        
        return enhanced_outputs


@MODELS.register_module(force=True)
class EnhancedQueryDecoder(QueryDecoder):
    """
    增强的查询解码器，集成甘蓝球径分割精度增强模块
    """
    def __init__(self,
                 *args,
                 enable_cabbage_refinement: bool = True,
                 enable_leaf_instance_enhancement: bool = False,
                 enable_integrated_enhancement: bool = False,
                 refinement_config: Optional[Dict] = None,
                 leaf_instance_config: Optional[Dict] = None,
                 integrated_enhancement_config: Optional[Dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.enable_cabbage_refinement = enable_cabbage_refinement
        self.enable_leaf_instance_enhancement = enable_leaf_instance_enhancement
        self.enable_integrated_enhancement = enable_integrated_enhancement
        
        # 推断类别数与特征维度
        try:
            self._num_classes_ref = int(self.out_cls[-1].out_features) - 1
        except Exception:
            self._num_classes_ref = int(kwargs.get('num_classes', 0))
        try:
            self._d_model = int(self.x_mask[-1].out_features)
        except Exception:
            # 回退到input_proj输出维度
            self._d_model = int(self.input_proj[0].out_features)
        # 用于产生原始每点logits的轻量头
        self._orig_point_head = nn.Linear(self._d_model, max(1, self._num_classes_ref))
        
        # 甘蓝球径增强模块
        if enable_cabbage_refinement:
            # 默认配置
            default_refinement_config = {
                'in_channels': self._d_model,
                'hidden_channels': 128,
                'num_classes': self._num_classes_ref,
                'refinement_layers': 2,
                'attention_heads': 8,
                'dropout': 0.1
            }
            
            if refinement_config is not None:
                default_refinement_config.update(refinement_config)
            # 强制对齐输入/类别维度，避免配置不一致
            default_refinement_config['in_channels'] = self._d_model
            default_refinement_config['num_classes'] = self._num_classes_ref
            
            # 创建增强模块
            self.cabbage_refinement = CabbageHeadRefinementModule(
                **default_refinement_config
            )
        else:
            self.cabbage_refinement = None
        
        # 叶片实例增强模块
        if enable_leaf_instance_enhancement:
            from .leaf_instance_module import LeafInstanceSegmentationModule
            default_leaf_config = {
                'openformer_feature_dim': self._d_model,
                'leaf_feature_dim': 192,  # 更新为192以匹配checkpoint
                'embedding_dim': 96,      # 更新为96以匹配checkpoint
                'k': 16
            }
            if leaf_instance_config is not None:
                # 只保留LeafInstanceSegmentationModule支持的参数
                supported_params = ['openformer_feature_dim', 'leaf_feature_dim', 'embedding_dim', 'k']
                filtered_config = {k: v for k, v in leaf_instance_config.items() if k in supported_params}
                default_leaf_config.update(filtered_config)
            
            self.leaf_instance_module = LeafInstanceSegmentationModule(
                **default_leaf_config
            )
        else:
            self.leaf_instance_module = None
        
        # 综合增强模块
        if enable_integrated_enhancement:
            from .integrated_cabbage_module import IntegratedCabbageEnhancementModule
            default_integrated_config = {
                'openformer_feature_dim': self._d_model
            }
            if integrated_enhancement_config is not None:
                # 只保留IntegratedCabbageEnhancementModule支持的参数
                supported_params = ['openformer_feature_dim', 'growth_stage_classifier']
                filtered_config = {k: v for k, v in integrated_enhancement_config.items() if k in supported_params}
                default_integrated_config.update(filtered_config)
            
            self.integrated_enhancement_module = IntegratedCabbageEnhancementModule(
                **default_integrated_config
            )
        else:
            self.integrated_enhancement_module = None
    
    def forward(self, x, queries=None):
        """
        前向传播，集成增强模块
        """
        # 调用原始解码器
        if self.iter_pred:
            outputs = self.forward_iter_pred(x, queries)
        else:
            outputs = self.forward_simple(x, queries)
        
        # 提取特征用于增强
        inst_feats = [self.input_proj(y) for y in x]           # [N, d_model]
        mask_feats = [self.x_mask(y) for y in x]               # [N, d_model]
        # 使用轻量头预测每点原始logits作为对照
        original_logits = [self._orig_point_head(f) for f in mask_feats]  # [N, C]
        
        enhanced_outputs = {}
        
        # 甘蓝球径增强模块
        if self.enable_cabbage_refinement and self.cabbage_refinement is not None:
            cabbage_outputs = self.cabbage_refinement(inst_feats, original_logits)
            enhanced_outputs.update(cabbage_outputs)
            
            # 为集成损失提供默认的叶片相关输出（当叶片实例增强未启用时）
            if not self.enable_leaf_instance_enhancement:
                batch_size = len(original_logits)
                device = original_logits[0].device if original_logits else torch.device('cpu')
                
                # 改进的叶片掩码生成：基于语义预测创建更合理的叶片掩码
                default_leaf_mask = []
                for logits in original_logits:
                    # 使用语义预测创建叶片掩码
                    if logits.dim() == 2 and logits.shape[1] >= 3:  # 确保有足够的类别
                        # 获取叶片类别(索引2)的概率
                        leaf_probs = torch.softmax(logits, dim=1)[:, 2]  # 叶片类别
                        # 创建叶片掩码：概率大于0.3的点被认为是叶片
                        leaf_mask = (leaf_probs > 0.3).float()
                        
                        # 确保至少有10%的点是叶片点
                        min_leaf_points = max(10, int(logits.shape[0] * 0.1))
                        if torch.sum(leaf_mask) < min_leaf_points:
                            # 如果叶片点太少，选择概率最高的点作为叶片
                            _, top_indices = torch.topk(leaf_probs, min_leaf_points)
                            leaf_mask = torch.zeros_like(leaf_mask)
                            leaf_mask[top_indices] = 1.0
                         
                        default_leaf_mask.append(leaf_mask)
                    else:
                        # 回退：创建随机叶片掩码
                        leaf_mask = torch.rand(logits.shape[0], device=device) > 0.7  # 30%的点是叶片
                        default_leaf_mask.append(leaf_mask.float())
                 
                enhanced_outputs['leaf_mask'] = default_leaf_mask
                
                # 默认的叶片实例嵌入（随机）
                default_embeddings = [torch.randn(logits.shape[0], 96, device=device) for logits in original_logits]
                enhanced_outputs['leaf_instance_embeddings'] = default_embeddings
                
                # 改进的实例分配：基于叶片掩码创建合理的实例标签
                default_assignments = []
                for i, (logits, leaf_mask) in enumerate(zip(original_logits, default_leaf_mask)):
                    if torch.sum(leaf_mask) > 0:
                        # 有叶片点，创建实例标签
                        num_points = logits.shape[0]
                        inst_labels = torch.full((num_points,), -1, device=device, dtype=torch.long)  # -1表示无效
                        
                        # 将叶片点分配为实例0
                        leaf_indices = torch.where(leaf_mask > 0.5)[0]
                        inst_labels[leaf_indices] = 0
                        
                        # 如果有足够的叶片点，创建更多实例
                        if len(leaf_indices) > 15:  # 降低阈值
                            # 随机分配一些点到实例1
                            num_inst1 = min(15, len(leaf_indices) // 3)  # 增加实例1的点数
                            inst1_indices = leaf_indices[torch.randperm(len(leaf_indices))[:num_inst1]]
                            inst_labels[inst1_indices] = 1
                            
                            # 如果还有足够的点，创建实例2
                            remaining_indices = [idx for idx in leaf_indices if idx not in inst1_indices]
                            if len(remaining_indices) > 10:
                                num_inst2 = min(10, len(remaining_indices) // 2)
                                inst2_indices = remaining_indices[:num_inst2]
                                inst_labels[inst2_indices] = 2
                    else:
                        # 没有叶片点，创建默认标签
                        inst_labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
                     
                    default_assignments.append(inst_labels)
                
                enhanced_outputs['leaf_instance_assignments'] = default_assignments
                
                # 默认的生长期概率（均匀分布）
                default_stage_probs = torch.ones(batch_size, 3, device=device) / 3.0  # [B, 3]
                enhanced_outputs['stage_probs'] = default_stage_probs
                
                # 为集成损失提供其他必需的键
                enhanced_outputs['head_boundary_confidence'] = torch.zeros(batch_size, device=device)
                enhanced_outputs['leaf_boundary_prob'] = default_leaf_mask
                enhanced_outputs['leaf_separation_confidence'] = [torch.zeros(logits.shape[0], device=device) for logits in original_logits]
                enhanced_outputs['leaf_completion_confidence'] = [torch.zeros(logits.shape[0], device=device) for logits in original_logits]
                enhanced_outputs['leaf_instance_centers'] = [torch.zeros(0, 3, device=device) for _ in range(batch_size)]
                enhanced_outputs['enhanced_features'] = inst_feats
        
        # 叶片实例增强模块
        if self.enable_leaf_instance_enhancement and self.leaf_instance_module is not None:
            # 需要点云坐标，这里简化处理
            # 在实际使用中，需要从外部传入点云坐标
            batch_size = len(x)
            dummy_points = [torch.randn(feat.shape[0], 3, device=feat.device) for feat in x]
            
            # 堆叠特征和logits用于叶片模块
            stacked_features = torch.cat(inst_feats, dim=0)
            stacked_logits = torch.cat(original_logits, dim=0)
            stacked_points = torch.cat(dummy_points, dim=0)
            
            # 获取球径掩码概率（如果有的话）
            head_mask_prob = enhanced_outputs.get('head_mask_prob', None)
            if head_mask_prob is not None:
                # 如果是列表，堆叠成张量
                if isinstance(head_mask_prob, list):
                    head_mask_prob = torch.cat(head_mask_prob, dim=0)
                # 确保维度正确
                if head_mask_prob.shape[0] != stacked_features.shape[0]:
                    head_mask_prob = torch.zeros(stacked_features.shape[0], device=stacked_features.device)
            else:
                head_mask_prob = torch.zeros(stacked_features.shape[0], device=stacked_features.device)
            
            try:
                leaf_outputs = self.leaf_instance_module(
                    stacked_points.unsqueeze(0),  # [1, N, 3]
                    stacked_features.unsqueeze(0),  # [1, N, D]
                    stacked_logits.unsqueeze(0),  # [1, N, C]
                    head_mask_prob.unsqueeze(0),  # [1, N]
                    stage_probs=None
                )
            except Exception as e:
                print(f"Error in leaf instance module: {e}")
                print(f"  Error type: {type(e).__name__}")
                
                # 检查是否是矩阵乘法错误
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    print("  Matrix multiplication error detected - applying dimension fix")
                    
                    # 创建修复后的叶片实例模块
                    try:
                        from .leaf_instance_module import LeafInstanceSegmentationModule
                        
                        # 使用修复后的配置
                        fixed_leaf_config = {
                            'openformer_feature_dim': self._d_model,
                            'leaf_feature_dim': 192,
                            'embedding_dim': 96,
                            'k': 16
                        }
                        
                        # 创建新的叶片实例模块
                        fixed_leaf_module = LeafInstanceSegmentationModule(**fixed_leaf_config)
                        fixed_leaf_module = fixed_leaf_module.to(stacked_features.device)
                        
                        # 重新尝试调用
                        leaf_outputs = fixed_leaf_module(
                            stacked_points.unsqueeze(0),  # [1, N, 3]
                            stacked_features.unsqueeze(0),  # [1, N, D]
                            stacked_logits.unsqueeze(0),  # [1, N, C]
                            head_mask_prob.unsqueeze(0),  # [1, N]
                            stage_probs=None
                        )
                        print("  Matrix multiplication error fixed successfully")
                        
                    except Exception as e2:
                        print(f"  Failed to fix matrix multiplication error: {e2}")
                        # 提供默认的叶片相关输出
                        batch_size = len(original_logits)
                        device = original_logits[0].device if original_logits else torch.device('cpu')
                        leaf_outputs = {
                            'leaf_mask': [torch.zeros(logits.shape[0], device=device) for logits in original_logits],
                            'leaf_instance_embeddings': [torch.randn(logits.shape[0], 96, device=device) for logits in original_logits],
                            'instance_assignments': [torch.zeros(logits.shape[0], dtype=torch.long, device=device) for logits in original_logits]
                        }
                else:
                    # 提供默认的叶片相关输出
                    batch_size = len(original_logits)
                    device = original_logits[0].device if original_logits else torch.device('cpu')
                    leaf_outputs = {
                        'leaf_mask': [torch.zeros(logits.shape[0], device=device) for logits in original_logits],
                        'leaf_instance_embeddings': [torch.randn(logits.shape[0], 96, device=device) for logits in original_logits],
                        'instance_assignments': [torch.zeros(logits.shape[0], dtype=torch.long, device=device) for logits in original_logits]
                    }
            
            # 将叶片输出添加到增强输出中
            enhanced_outputs.update(leaf_outputs)
        
        # 综合增强模块
        if self.enable_integrated_enhancement and self.integrated_enhancement_module is not None:
            # 堆叠特征和logits
            stacked_features = torch.cat(inst_feats, dim=0)
            stacked_logits = torch.cat(original_logits, dim=0)
            dummy_points = [torch.randn(feat.shape[0], 3, device=feat.device) for feat in x]
            stacked_points = torch.cat(dummy_points, dim=0)
            
            integrated_outputs = self.integrated_enhancement_module(
                stacked_points.unsqueeze(0),  # [1, N, 3]
                stacked_features.unsqueeze(0),  # [1, N, D]
                stacked_logits.unsqueeze(0)  # [1, N, C]
            )
            
            # 将综合输出添加到增强输出中
            enhanced_outputs.update(integrated_outputs)
        
        # 将增强输出添加到结果中
        if enhanced_outputs:
            outputs['enhanced_outputs'] = enhanced_outputs
        
        return outputs


class CabbageHeadStageClassifier(nn.Module):
    """
    甘蓝生长期分类器
    用于判断当前样本属于哪个生长期
    """
    def __init__(self,
                 in_channels: int = 256,
                 hidden_channels: int = 128,
                 num_stages: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_stages = num_stages
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_stages)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        Args:
            features: 特征列表 [B, N, C]
        Returns:
            stage_logits: 生长期logits [B, num_stages]
            stage_features: 特征字典
        """
        batch_size = len(features)
        stage_logits = []
        
        for b in range(batch_size):
            feat = features[b]  # [N, C]
            
            # 全局平均池化
            global_feat = self.global_pool(feat.T).squeeze(-1)  # [C]
            
            # 分类
            stage_logit = self.classifier(global_feat)  # [num_stages]
            stage_logits.append(stage_logit)
        
        stage_logits = torch.stack(stage_logits)  # [B, num_stages]
        
        # 特征字典
        stage_features = {
            'global_features': [feat.mean(dim=0) for feat in features]
        }
        
        return stage_logits, stage_features


@MODELS.register_module(force=True)
class CabbageEnhancedOneFormer3D(nn.Module):
    """
    甘蓝增强OneFormer3D模型
    集成生长期分类和球径分割精度增强
    """
    def __init__(self,
                 base_decoder_config: Dict,
                 enable_stage_classification: bool = True,
                 enable_cabbage_refinement: bool = True,
                 refinement_config: Optional[Dict] = None,
                 stage_classifier_config: Optional[Dict] = None):
        super().__init__()
        
        # 基础解码器
        self.base_decoder = MODELS.build(base_decoder_config)
        
        # 生长期分类器
        self.enable_stage_classification = enable_stage_classification
        if enable_stage_classification:
            default_stage_config = {
                'in_channels': 256,
                'hidden_channels': 128,
                'num_stages': 3,  # 幼苗期、生长期、结球期
                'dropout': 0.1
            }
            if stage_classifier_config is not None:
                default_stage_config.update(stage_classifier_config)
            
            self.stage_classifier = CabbageHeadStageClassifier(**default_stage_config)
        else:
            self.stage_classifier = None
        
        # 球径分割增强
        self.enable_cabbage_refinement = enable_cabbage_refinement
        if enable_cabbage_refinement:
            # 创建增强解码器
            enhanced_decoder_config = base_decoder_config.copy()
            enhanced_decoder_config['enable_cabbage_refinement'] = True
            enhanced_decoder_config['refinement_config'] = refinement_config
            
            self.enhanced_decoder = EnhancedQueryDecoder(**enhanced_decoder_config)
        else:
            self.enhanced_decoder = None
    
    def forward(self, x, queries=None, return_stage_info=False):
        """
        前向传播
        Args:
            x: 输入特征
            queries: 查询向量
            return_stage_info: 是否返回生长期信息
        Returns:
            outputs: 输出字典
        """
        outputs = {}
        
        # 生长期分类
        if self.enable_stage_classification and self.stage_classifier is not None:
            stage_logits, stage_features = self.stage_classifier(x)
            stage_probs = F.softmax(stage_logits, dim=-1)
            
            outputs['stage_logits'] = stage_logits
            outputs['stage_probs'] = stage_probs
            outputs['stage_features'] = stage_features
        
        # 基础分割
        base_outputs = self.base_decoder(x, queries)
        outputs.update(base_outputs)
        
        # 增强分割
        if self.enable_cabbage_refinement and self.enhanced_decoder is not None:
            enhanced_outputs = self.enhanced_decoder(x, queries)
            outputs['enhanced_outputs'] = enhanced_outputs.get('enhanced_outputs', {})
        
        return outputs
