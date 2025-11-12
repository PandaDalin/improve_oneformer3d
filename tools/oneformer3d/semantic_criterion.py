import torch
import torch.nn.functional as F
import numpy as np

from mmdet3d.registry import MODELS


@MODELS.register_module(force=True)
class ScanNetSemanticCriterion:
    """Semantic criterion for ScanNet.

    Args:
        ignore_index (int): Ignore index.
        loss_weight (float): Loss weight.
    """

    def __init__(self, ignore_index, loss_weight):
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (dict): Predictions with List `sem_preds`
                of len batch_size, each of shape
                (n_queries_i, n_classes + 1).
            insts (list): Ground truth of len batch_size, 
                each InstanceData_ with `sp_masks` of shape
                (n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic loss value.
        """
        losses = []
        for pred_mask, gt_mask in zip(pred['sem_preds'], insts):
            if self.ignore_index >= 0:
                pred_mask = pred_mask[:, :-1]
            losses.append(F.cross_entropy(
                pred_mask,
                gt_mask.sp_masks.float().argmax(0),
                ignore_index=self.ignore_index))
        loss = self.loss_weight * torch.mean(torch.stack(losses))
        return dict(seg_loss=loss)


@MODELS.register_module(force=True)
class S3DISSemanticCriterion:
    """Semantic criterion for S3DIS.

    Args:
        loss_weight (float): loss weight.
        seg_loss (ConfigDict): loss config.
        class_weight (list[float], optional): 类别权重列表。
        class_weight_dict (dict, optional): 类别权重字典，格式为 {class_name: weight}。
        dynamic_weight (bool): 是否使用动态权重计算。
        weight_decay (float): 权重衰减因子，用于动态权重调整。
    """

    def __init__(self,
                 loss_weight,
                 seg_loss=dict(
                     type='mmdet.CrossEntropyLoss', use_sigmoid=True),
                 class_weight=None,
                 class_weight_dict=None,
                 dynamic_weight=False,
                 weight_decay=0.1):
        self.loss_weight = loss_weight
        self.dynamic_weight = dynamic_weight
        self.weight_decay = weight_decay
        
        # 处理类别权重
        if class_weight_dict is not None:
            # 使用类别名称字典
            self.class_names = ('corm', 'ground', 'leaf')
            class_weight = [class_weight_dict.get(name, 1.0) for name in self.class_names]
        elif class_weight is not None:
            # 直接使用权重列表
            self.class_names = ('corm', 'ground', 'leaf')
        else:
            self.class_names = ('corm', 'ground', 'leaf')
            class_weight = [1.0] * len(self.class_names)
        
        # 确保权重列表长度正确
        if len(class_weight) != len(self.class_names):
            raise ValueError(f"class_weight长度({len(class_weight)})与类别数量({len(self.class_names)})不匹配")
        
        self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
        self.class_weight_dict = {name: weight for name, weight in zip(self.class_names, class_weight)}
        
        # 构建损失函数
        if class_weight is not None:
            seg_loss = seg_loss.copy()
            seg_loss['class_weight'] = class_weight
        self.seg_loss = MODELS.build(seg_loss)
        
        # 初始化动态权重统计
        if self.dynamic_weight:
            self.class_loss_history = {name: [] for name in self.class_names}
            self.class_count_history = {name: [] for name in self.class_names}

    def update_dynamic_weights(self, pred_masks, gt_masks):
        """更新动态权重基于当前批次的损失分布"""
        if not self.dynamic_weight:
            return
            
        # 计算每个类别的平均损失
        class_losses = {name: [] for name in self.class_names}
        class_counts = {name: 0 for name in self.class_names}
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            try:
                # 安全地获取gt_labels
                if hasattr(gt_mask, 'sp_masks'):
                    sp_masks = gt_mask.sp_masks
                    if isinstance(sp_masks, torch.Tensor):
                        # 确保sp_masks是正确的形状
                        if sp_masks.dim() == 2:
                            # 如果是(classes, points)形状
                            if sp_masks.shape[0] <= sp_masks.shape[1]:
                                gt_labels = sp_masks.float().argmax(0)
                            else:
                                gt_labels = sp_masks.float().argmax(-1)
                        else:
                            # 如果是一维张量，直接使用
                            gt_labels = sp_masks.long()
                    else:
                        # 如果不是张量，转换为张量
                        sp_masks = torch.tensor(sp_masks, dtype=torch.float32)
                        if sp_masks.dim() == 2:
                            gt_labels = sp_masks.argmax(0)
                        else:
                            gt_labels = sp_masks.long()
                else:
                    # 如果没有sp_masks属性，跳过这个样本
                    continue
                
                # 确保pred_mask和gt_labels的形状匹配
                if pred_mask.shape[0] != gt_labels.shape[0]:
                    # 如果形状不匹配，跳过这个样本
                    continue
                
                # 计算每个类别的损失
                for class_idx, class_name in enumerate(self.class_names):
                    class_mask = (gt_labels == class_idx)
                    if class_mask.sum() > 0:
                        try:
                            class_pred = pred_mask[class_mask, :]
                            class_gt = gt_labels[class_mask]
                            class_loss = F.cross_entropy(class_pred, class_gt, reduction='mean')
                            class_losses[class_name].append(class_loss.item())
                            class_counts[class_name] += class_mask.sum().item()
                        except Exception as e:
                            # 如果出现错误，跳过这个类别
                            continue
                            
            except Exception as e:
                # 如果处理整个样本时出现错误，跳过这个样本
                continue
        
        # 更新历史记录
        for class_name in self.class_names:
            if class_losses[class_name]:
                avg_loss = np.mean(class_losses[class_name])
                self.class_loss_history[class_name].append(avg_loss)
                self.class_count_history[class_name].append(class_counts[class_name])
                
                # 保持历史记录长度
                if len(self.class_loss_history[class_name]) > 100:
                    self.class_loss_history[class_name] = self.class_loss_history[class_name][-50:]
                    self.class_count_history[class_name] = self.class_count_history[class_name][-50:]
        
        # 计算新的动态权重
        if all(len(history) > 0 for history in self.class_loss_history.values()):
            avg_losses = {name: np.mean(history) for name, history in self.class_loss_history.items()}
            max_loss = max(avg_losses.values())
            
            # 基于损失比例计算权重
            new_weights = {}
            for class_name in self.class_names:
                if avg_losses[class_name] > 0:
                    # 损失越高的类别获得更高的权重
                    new_weight = max_loss / avg_losses[class_name]
                    # 应用权重衰减
                    current_weight = self.class_weight_dict[class_name]
                    new_weights[class_name] = (1 - self.weight_decay) * current_weight + self.weight_decay * new_weight
                else:
                    new_weights[class_name] = self.class_weight_dict[class_name]
            
            # 更新权重
            self.class_weight_dict = new_weights
            self.class_weight = torch.tensor([new_weights[name] for name in self.class_names], dtype=torch.float32)
            
            # 更新损失函数的权重
            if hasattr(self.seg_loss, 'class_weight'):
                self.seg_loss.class_weight = self.class_weight

    def get_layer_loss(self, layer, aux_outputs, insts):
        """Calculate loss at intermediate level.

        Args:
            layer (int): transformer layer number
            aux_outputs (dict): Predictions with List `masks`
                of len batch_size, each of shape
                (n_points_i, n_classes + 1).
            insts (list): Ground truth of len batch_size, 
                each InstanceData_ with `sp_masks` of shape
                (n_classes + 1, n_points_i).

        Returns:
            Dict: with semantic loss value.
        """
        pred_masks = aux_outputs['masks']
        
        # 更新动态权重
        if self.dynamic_weight:
            self.update_dynamic_weights(pred_masks, insts)
        
        seg_losses = []
        for pred_mask, gt_mask in zip(pred_masks, insts):
            seg_loss = self.seg_loss(
                pred_mask.T, gt_mask.sp_masks.float().argmax(0))
            seg_losses.append(seg_loss)

        seg_loss = self.loss_weight * torch.mean(torch.stack(seg_losses))
        return {f'layer_{layer}_seg_loss': seg_loss}

    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (dict): Predictions with List `masks`
                of len batch_size, each of shape
                (n_points_i, n_classes + 1).
            insts (list): Ground truth of len batch_size, 
                each InstanceData_ with `sp_masks` of shape
                (n_classes + 1, n_points_i).

        Returns:
            Dict: with semantic loss value.
        """
        pred_masks = pred['masks']
        
        # 更新动态权重
        if self.dynamic_weight:
            self.update_dynamic_weights(pred_masks, insts)
        
        seg_losses = []
        for pred_mask, gt_mask in zip(pred_masks, insts):
            seg_loss = self.seg_loss(
                pred_mask.T, gt_mask.sp_masks.float().argmax(0))
            seg_losses.append(seg_loss)

        seg_loss = self.loss_weight * torch.mean(torch.stack(seg_losses))
        loss = {'last_layer_seg_loss': seg_loss}
        
        # 添加类别权重信息到损失字典中（用于监控）
        if self.dynamic_weight:
            for class_name, weight in self.class_weight_dict.items():
                loss[f'class_weight_{class_name}'] = torch.tensor(weight, dtype=torch.float32)

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i = self.get_layer_loss(i, aux_outputs, insts)
                loss.update(loss_i)

        return loss

    def get_class_weights(self):
        """获取当前类别权重"""
        return self.class_weight_dict.copy()

    def set_class_weights(self, class_weight_dict):
        """设置类别权重"""
        for class_name, weight in class_weight_dict.items():
            if class_name in self.class_names:
                self.class_weight_dict[class_name] = weight
        
        self.class_weight = torch.tensor([self.class_weight_dict[name] for name in self.class_names], dtype=torch.float32)
        
        # 更新损失函数的权重
        if hasattr(self.seg_loss, 'class_weight'):
            self.seg_loss.class_weight = self.class_weight
