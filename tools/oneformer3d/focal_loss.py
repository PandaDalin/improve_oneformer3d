import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module(force=True)
class CabbageSoftmaxFocalLoss(nn.Module):
    """
    Softmax-style Focal Loss for 3-class segmentation.
    """

    def __init__(self,
                 num_classes=3,
                 gamma=2.0,
                 class_weight=None,
                 ignore_index=255,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-8):
        super().__init__()
        assert num_classes == 3, "This loss is configured for 3 classes (corm/ground/leaf)."
        self.num_classes = num_classes
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)

        if class_weight is not None:
            assert len(class_weight) == num_classes, \
                f"class_weight length {len(class_weight)} != num_classes {num_classes}"
            self.register_buffer('class_weight',
                                 torch.tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None

    def forward(self, logits, target, avg_factor=None):
        """
        Args:
            logits: Tensor of shape (N, C) or (B, N, C) or (C, N). **Logits** (not probabilities).
            target: Long tensor of shape (N,) or (B, N) with values in {0,1,2} or ignore_index.
            avg_factor: Optional averaging factor when reduction='mean'.

        Returns:
            Tensor: focal loss (scalar if reduced, else per-sample)
        """
        # Align devices
        if logits.device != target.device:
            target = target.to(logits.device)

        # Handle different input shapes
        if logits.dim() == 3:          # (B, N, C)
            B, N, C = logits.shape
            logits = logits.reshape(-1, C)
            target = target.reshape(-1)
        elif logits.dim() == 2:        # (N, C) or (C, N)
            if logits.shape[0] == self.num_classes:  # (C, N) -> transpose to (N, C)
                logits = logits.t()  # (N, C)
            N, C = logits.shape
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")

        # Sanity checks
        if C != self.num_classes:
            raise ValueError(f"logits.shape[-1]={C} != num_classes={self.num_classes}")

        # Ensure target has the same number of points as logits
        if target.shape[0] != logits.shape[0]:
            # If target has more points, truncate it
            if target.shape[0] > logits.shape[0]:
                target = target[:logits.shape[0]]
            # If target has fewer points, pad it with ignore_index
            else:
                pad_size = logits.shape[0] - target.shape[0]
                target = torch.cat([target, torch.full((pad_size,), self.ignore_index, device=target.device, dtype=target.dtype)])

        # Mask out ignore_index
        valid_mask = (target != self.ignore_index)
        
        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)

        logits = logits[valid_mask]          # (M, C)
        target = target[valid_mask].long()   # (M,)

        # Standard CE on log-softmax (safer than softmax + log)
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()  # (M, C)

        # p_t = P(model predicts the true class)
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(self.eps)  # (M,)

        # focal modulation
        focal_weight = (1.0 - pt).pow(self.gamma)  # (M,)

        # class-weighted NLL loss
        if self.class_weight is not None:
            # Ensure class_weight is on the same device as log_prob
            class_weight = self.class_weight.to(log_prob.device)
            ce = F.nll_loss(
                log_prob, target,
                weight=class_weight, reduction='none'
            )  # (M,)
        else:
            ce = F.nll_loss(
                log_prob, target, reduction='none'
            )  # (M,)

        loss = focal_weight * ce  # (M,)

        # Reduction
        if self.reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / float(avg_factor)
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return self.loss_weight * loss


@MODELS.register_module(force=True)
class S3DISCabbageFocalCriterion:
    """
    S3DIS Cabbage Focal Loss Criterion
    """

    def __init__(self,
                 loss_weight=1.0,
                 focal_loss=dict(
                     type='CabbageSoftmaxFocalLoss',
                     num_classes=3,
                     gamma=2.0,
                     class_weight=[3.0, 1.0, 1.0],
                     ignore_index=255,
                     reduction='mean',
                     loss_weight=1.0)):
        self.focal_loss = MODELS.build(focal_loss)
        self.loss_weight = float(loss_weight)
        self.num_classes = 3
        self.ignore_index = getattr(self.focal_loss, 'ignore_index', 255)

    @staticmethod
    def _extract_gt_labels(gt_mask, device):
        """
        Robustly extract per-point semantic labels as a Long tensor (N,).
        """
        if hasattr(gt_mask, 'pts_semantic_mask') and gt_mask.pts_semantic_mask is not None:
            gt_labels = torch.as_tensor(gt_mask.pts_semantic_mask, device=device)
        elif hasattr(gt_mask, 'sp_masks') and gt_mask.sp_masks is not None:
            sp = gt_mask.sp_masks
            if isinstance(sp, torch.Tensor):
                sp = sp.to(device)
            else:
                sp = torch.as_tensor(sp, device=device)
            if sp.dim() == 2 and sp.shape[0] <= sp.shape[1]:
                gt_labels = sp.float().argmax(0)
            else:
                gt_labels = sp.float().argmax(-1)
        elif hasattr(gt_mask, 'pts_instance_mask') and gt_mask.pts_instance_mask is not None:
            gt_labels = torch.as_tensor(gt_mask.pts_instance_mask, device=device)
        else:
            raise ValueError("Cannot extract semantic labels from gt_mask.")

        return gt_labels.long()

    def maybe_slice_num_classes(self, pred_mask):
        """
        If upstream model still outputs C > 3, take only the first 3 channels.
        """
        if pred_mask.shape[1] > self.num_classes:
            pred_mask = pred_mask[:, :self.num_classes]
        return pred_mask

    def _loss_single(self, pred_mask, gt_mask):
        """
        Compute loss for a single sample.
        """
        pred_mask = self.maybe_slice_num_classes(pred_mask)
        device = pred_mask.device

        # Extract GT labels
        gt_labels = self._extract_gt_labels(gt_mask, device=device)

        invalid = (gt_labels < 0) | (gt_labels >= self.num_classes)
        if torch.any(invalid):
            gt_labels = gt_labels.clone()
            gt_labels[invalid] = self.ignore_index
            
        return self.focal_loss(pred_mask, gt_labels)

    def get_layer_loss(self, layer, aux_outputs, insts):
        pred_masks = aux_outputs['masks']
        
        per_item = []
        for pm, gt in zip(pred_masks, insts):
            try:
                loss = self._loss_single(pm, gt)
                per_item.append(loss)
            except Exception:
                per_item.append(pm.new_tensor(0.0))
                
        if len(per_item) == 0:
            return {f'layer_{layer}_focal_loss': torch.tensor(0.0, device=pred_masks[0].device)}
            
        loss = torch.stack(per_item).mean() * self.loss_weight
        return {f'layer_{layer}_focal_loss': loss}

    def __call__(self, pred, insts):
        """
        Main loss computation
        """
        pred_masks = pred['masks']
        
        per_item = []
        for pm, gt in zip(pred_masks, insts):
            try:
                loss = self._loss_single(pm, gt)
                per_item.append(loss)
            except Exception:
                per_item.append(pm.new_tensor(0.0))

        if len(per_item) == 0:
            device = pred_masks[0].device if pred_masks else 'cpu'
            main_loss = torch.tensor(0.0, device=device)
        else:
            main_loss = torch.stack(per_item).mean() * self.loss_weight

        loss = {'last_layer_focal_loss': main_loss}

        # Aux heads
        if 'aux_outputs' in pred and isinstance(pred['aux_outputs'], (list, tuple)):
            for i, aux in enumerate(pred['aux_outputs']):
                loss.update(self.get_layer_loss(i, aux, insts))

        return loss
