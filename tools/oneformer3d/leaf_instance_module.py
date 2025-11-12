import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


class LeafInstanceSegmentationModule(nn.Module):
    """
    叶片实例分割增强模块
    专门解决叶片之间的残缺和粘连问题
    """
    def __init__(self,
                 openformer_feature_dim: int = 256,
                 leaf_feature_dim: int = 192,  # 更新为192以匹配checkpoint
                 embedding_dim: int = 96,      # 更新为96以匹配checkpoint
                 k: int = 16):
        super().__init__()
        self.openformer_feature_dim = openformer_feature_dim
        self.leaf_feature_dim = leaf_feature_dim
        self.embedding_dim = embedding_dim
        self.k = k
        self.leaf_feature_extractor = LeafFeatureExtractor(openformer_feature_dim, leaf_feature_dim, k)
        self.leaf_boundary_detector = LeafBoundaryDetector(leaf_feature_dim, k)
        self.instance_embedding_net = InstanceEmbeddingNetwork(leaf_feature_dim, embedding_dim)
        self.leaf_separation_net = LeafSeparationNetwork(leaf_feature_dim, k)
        self.leaf_completion_net = LeafCompletionNetwork(leaf_feature_dim, k)
        self.instance_consistency_net = InstanceConsistencyNetwork(embedding_dim)

    def forward(self,
                points: torch.Tensor,
                openformer_features: torch.Tensor,
                openformer_logits: torch.Tensor,
                head_mask_prob: torch.Tensor,
                stage_probs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, num_points = points.shape[:2]
        leaf_mask = self._extract_leaf_mask(openformer_logits, head_mask_prob)
        leaf_features = self.leaf_feature_extractor(points, openformer_features, leaf_mask)
        boundary_info = self.leaf_boundary_detector(points, leaf_features, leaf_mask)
        separated_features = self.leaf_separation_net(points, leaf_features, boundary_info, leaf_mask)
        completed_features = self.leaf_completion_net(points, separated_features, leaf_mask, stage_probs)
        instance_embeddings = self.instance_embedding_net(completed_features, leaf_mask)
        refined_embeddings = self.instance_consistency_net(points, instance_embeddings, leaf_mask)
        instance_results = self._generate_instance_predictions(points, refined_embeddings, leaf_mask, boundary_info)
        outputs = {
            'leaf_mask': leaf_mask,
            'leaf_features': completed_features,
            'boundary_prob': boundary_info['boundary_prob'],
            'instance_embeddings': refined_embeddings,
            'instance_centers': instance_results['centers'],
            'instance_assignments': instance_results['assignments'],
            'separation_confidence': boundary_info.get('separation_confidence', None),
            'completion_confidence': self._compute_completion_confidence(completed_features)
        }
        return outputs

    def _extract_leaf_mask(self, openformer_logits: torch.Tensor, head_mask_prob: torch.Tensor) -> torch.Tensor:
        leaf_prob = F.softmax(openformer_logits, dim=-1)[:, :, -1]
        leaf_mask = (leaf_prob > 0.3) & (head_mask_prob < 0.3)
        return leaf_mask.float()

    def _generate_instance_predictions(self,
                                       points: torch.Tensor,
                                       embeddings: torch.Tensor,
                                       leaf_mask: torch.Tensor,
                                       boundary_info: Dict) -> Dict:
        batch_size = points.shape[0]
        instance_centers = []
        instance_assignments = []
        for b in range(batch_size):
            if torch.sum(leaf_mask[b]) < 10:
                centers = torch.empty(0, 3, device=points.device)
                assignments = torch.zeros(points.shape[1], dtype=torch.long, device=points.device)
            else:
                centers, assignments = self._cluster_leaf_instances(points[b], embeddings[b], leaf_mask[b], boundary_info['boundary_prob'][b])
            instance_centers.append(centers)
            instance_assignments.append(assignments)
        return {'centers': instance_centers, 'assignments': instance_assignments}

    def _cluster_leaf_instances(self,
                                points: torch.Tensor,
                                embeddings: torch.Tensor,
                                leaf_mask: torch.Tensor,
                                boundary_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        leaf_points = points[leaf_mask.bool()]
        leaf_embeddings = embeddings[leaf_mask.bool()]
        if len(leaf_points) < 10:
            return torch.empty(0, 3, device=points.device), torch.zeros(len(points), dtype=torch.long, device=points.device)
        centers, assignments = self._adaptive_clustering(leaf_points, leaf_embeddings, boundary_prob[leaf_mask.bool()])
        full_assignments = torch.zeros(len(points), dtype=torch.long, device=points.device)
        full_assignments[leaf_mask.bool()] = assignments
        return centers, full_assignments

    def _adaptive_clustering(self,
                              points: torch.Tensor,
                              embeddings: torch.Tensor,
                              boundary_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_points = len(points)
        embedding_dist = torch.cdist(embeddings, embeddings)
        spatial_dist = torch.cdist(points, points)
        embedding_dist = embedding_dist / (torch.max(embedding_dist) + 1e-8)
        spatial_dist = spatial_dist / (torch.max(spatial_dist) + 1e-8)
        combined_dist = 0.7 * embedding_dist + 0.3 * spatial_dist
        threshold = 0.3
        visited = torch.zeros(num_points, dtype=torch.bool, device=points.device)
        assignments = torch.zeros(num_points, dtype=torch.long, device=points.device)
        centers = []
        cluster_id = 1
        for i in range(num_points):
            if visited[i] or boundary_prob[i] > 0.7:
                continue
            similar_mask = combined_dist[i] < threshold
            cluster_points = points[similar_mask]
            if torch.sum(similar_mask) >= 5:
                cluster_center = torch.mean(cluster_points, dim=0)
                centers.append(cluster_center)
                assignments[similar_mask] = cluster_id
                visited[similar_mask] = True
                cluster_id += 1
        if len(centers) > 0:
            centers = torch.stack(centers)
        else:
            centers = torch.empty(0, 3, device=points.device)
        return centers, assignments

    def _compute_completion_confidence(self, features: torch.Tensor) -> torch.Tensor:
        feature_var = torch.var(features, dim=-1)
        confidence = torch.sigmoid(-feature_var)
        return confidence


class LeafFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, k: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.leaf_geometry_encoder = nn.Sequential(
            nn.Linear(input_dim + 6, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.leaf_texture_encoder = LeafTextureEncoder(k)

    def forward(self,
                points: torch.Tensor,
                features: torch.Tensor,
                leaf_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_points = points.shape[:2]
        enhanced_features = []
        for b in range(batch_size):
            # 确保特征维度匹配
            current_features = features[b]
            
            # 检查特征张量的形状 - 确保是 (num_points, feature_dim)
            if current_features.dim() != 2:
                current_features = current_features.view(num_points, -1)
            
            # 确保特征张量有正确的点数
            if current_features.shape[0] != num_points:
                # 如果点数不匹配，创建正确大小的零张量
                current_features = torch.zeros(num_points, current_features.shape[-1], device=current_features.device)
            
            # 如果特征维度不匹配，进行线性变换
            if current_features.shape[-1] != self.input_dim:
                # 如果特征维度不匹配，进行线性变换
                if not hasattr(self, 'feature_proj'):
                    self.feature_proj = nn.Linear(current_features.shape[-1], self.input_dim).to(current_features.device)
                current_features = self.feature_proj(current_features)
            
            # 计算leaf_geom，确保维度正确
            leaf_geom = self._compute_leaf_geometry(points[b], leaf_mask[b])
            _ = self.leaf_texture_encoder(points[b], leaf_mask[b])
            
            # 确保leaf_geom的维度与current_features匹配
            if leaf_geom.shape[0] != current_features.shape[0]:
                # 创建正确大小的零张量
                leaf_geom = torch.zeros(current_features.shape[0], 6, device=current_features.device)
            
            # 确保combined张量的维度正确
            if current_features.shape[0] != leaf_geom.shape[0]:
                # 如果还是有问题，使用零张量
                leaf_geom = torch.zeros(current_features.shape[0], 6, device=current_features.device)
            
            # 确保两个张量都是2D的
            if current_features.dim() != 2:
                current_features = current_features.view(current_features.shape[0], -1)
            if leaf_geom.dim() != 2:
                leaf_geom = leaf_geom.view(leaf_geom.shape[0], -1)
            
            # 最终检查：确保combined张量的维度正确
            if current_features.shape[-1] != self.input_dim:
                # 如果特征维度仍然不匹配，使用零张量
                current_features = torch.zeros(current_features.shape[0], self.input_dim, device=current_features.device)
            
            combined = torch.cat([current_features, leaf_geom], dim=-1)
            encoded = self.leaf_geometry_encoder(combined)
            enhanced_features.append(encoded)
        return torch.stack(enhanced_features, dim=0)

    def _compute_leaf_geometry(self,
                               points: torch.Tensor,
                               leaf_mask: torch.Tensor) -> torch.Tensor:
        num_points = points.shape[0]
        leaf_features = torch.zeros(num_points, 6, device=points.device)
        
        # 确保leaf_mask是布尔类型
        if leaf_mask.dtype != torch.bool:
            leaf_mask = leaf_mask.bool()
        
        if torch.sum(leaf_mask) == 0:
            return leaf_features
        
        leaf_points = points[leaf_mask]
        if len(leaf_points) == 0:
            return leaf_features
            
        leaf_center = torch.mean(leaf_points, dim=0)
        
        for i in range(num_points):
            point = points[i]
            dist_to_leaf_center = torch.norm(point - leaf_center)
            relative_height = point[2] - leaf_center[2]
            horizontal_dist = torch.norm(point[:2] - leaf_center[:2])
            local_density = self._compute_local_density(points, i, leaf_mask)
            local_curvature = self._compute_leaf_curvature(points, i)
            radial_direction = torch.atan2(point[1] - leaf_center[1], point[0] - leaf_center[0])
            
            leaf_features[i] = torch.tensor([
                dist_to_leaf_center,
                relative_height,
                horizontal_dist,
                local_density,
                local_curvature,
                radial_direction
            ], device=points.device)
        
        # 确保返回的张量维度正确
        if leaf_features.shape[0] != num_points:
            # 如果维度不匹配，创建正确大小的零张量
            leaf_features = torch.zeros(num_points, 6, device=points.device)
        
        # 确保返回的是2D张量
        if leaf_features.dim() != 2:
            leaf_features = leaf_features.view(num_points, 6)
        
        return leaf_features

    def _compute_local_density(self,
                               points: torch.Tensor,
                               center_idx: int,
                               leaf_mask: torch.Tensor) -> torch.Tensor:
        center = points[center_idx]
        distances = torch.norm(points - center, dim=1)
        
        # 确保leaf_mask是布尔类型
        if leaf_mask.dtype != torch.bool:
            leaf_mask = leaf_mask.bool()
            
        neighbors_in_radius = torch.sum((distances < 0.02) & leaf_mask)
        return neighbors_in_radius.float()

    def _compute_leaf_curvature(self, points: torch.Tensor, center_idx: int) -> torch.Tensor:
        center = points[center_idx]
        distances = torch.norm(points - center, dim=1)
        k = min(10, len(points))
        _, neighbor_indices = torch.topk(distances, k, largest=False)
        neighbors = points[neighbor_indices]
        
        if len(neighbors) < 6:
            return torch.tensor(0.0, device=points.device)
            
        centered_neighbors = neighbors - center
        
        try:
            # 确保矩阵维度正确
            if centered_neighbors.shape[0] < 3:
                return torch.tensor(0.0, device=points.device)
                
            # 计算协方差矩阵
            cov_matrix = torch.mm(centered_neighbors.T, centered_neighbors) / len(neighbors)
            
            # 确保协方差矩阵是3x3的
            if cov_matrix.shape[0] != 3 or cov_matrix.shape[1] != 3:
                return torch.tensor(0.0, device=points.device)
                
            eigenvalues, _ = torch.linalg.eigh(cov_matrix)
            
            # 确保有足够的特征值
            if len(eigenvalues) < 3:
                return torch.tensor(0.0, device=points.device)
                
            curvature = eigenvalues[0] / (eigenvalues[2] + 1e-8)
            return curvature
            
        except Exception as e:
            return torch.tensor(0.0, device=points.device)


class LeafTextureEncoder(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    def forward(self, points: torch.Tensor, leaf_mask: torch.Tensor) -> torch.Tensor:
        num_points = points.shape[0]
        texture_features = torch.zeros(num_points, 32, device=points.device)
        return texture_features


class LeafBoundaryDetector(nn.Module):
    def __init__(self, feature_dim: int, k: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.boundary_encoder = nn.Sequential(
            nn.Linear(feature_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self,
                points: torch.Tensor,
                features: torch.Tensor,
                leaf_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, num_points = points.shape[:2]
        boundary_probs = []
        separation_confidences = []
        for b in range(batch_size):
            boundary_prob = self._detect_leaf_boundaries(points[b], features[b], leaf_mask[b])
            boundary_probs.append(boundary_prob)
            separation_conf = self._compute_separation_confidence(points[b], boundary_prob, leaf_mask[b])
            separation_confidences.append(separation_conf)
        return {
            'boundary_prob': torch.stack(boundary_probs, dim=0),
            'boundary_features': features,
            'separation_confidence': torch.stack(separation_confidences, dim=0)
        }

    def _detect_leaf_boundaries(self,
                                points: torch.Tensor,
                                features: torch.Tensor,
                                leaf_mask: torch.Tensor) -> torch.Tensor:
        num_points = points.shape[0]
        boundary_scores = torch.zeros(num_points, device=points.device)
        if torch.sum(leaf_mask) < 10:
            return boundary_scores
        
        # 确保特征维度匹配
        expected_input_dim = self.feature_dim + 3  # feature_dim + 3 (spatial features)
        
        # 检查特征张量的形状 - 确保保持 (num_points, feature_dim) 的形状
        if features.dim() != 2:
            features = features.view(features.shape[0], -1)
        
        # 确保特征张量有正确的点数
        if features.shape[0] != num_points:
            # 如果点数不匹配，创建正确大小的零张量
            features = torch.zeros(num_points, features.shape[-1], device=features.device)
        
        # 如果特征维度不匹配，进行线性变换
        if features.shape[-1] != self.feature_dim:
            if not hasattr(self, 'feature_proj'):
                self.feature_proj = nn.Linear(features.shape[-1], self.feature_dim).to(features.device)
            features = self.feature_proj(features)
        
        for i in range(num_points):
            if not bool(leaf_mask[i]):
                continue
            center = points[i]
            distances = torch.norm(points - center, dim=1)
            _, neighbor_indices = torch.topk(distances, min(self.k, num_points), largest=False)
            center_feat = features[i]
            neighbor_feats = features[neighbor_indices]
            feature_variance = torch.var(torch.norm(neighbor_feats - center_feat, dim=1))
            spatial_feat = torch.cat([center, feature_variance.unsqueeze(0)])
            
            combined_feat = torch.cat([center_feat, spatial_feat])
            
            # 确保输入维度正确
            if combined_feat.shape[0] != expected_input_dim:
                # 如果维度不匹配，进行填充或截断
                if combined_feat.shape[0] < expected_input_dim:
                    padding = torch.zeros(expected_input_dim - combined_feat.shape[0], device=combined_feat.device)
                    combined_feat = torch.cat([combined_feat, padding])
                else:
                    combined_feat = combined_feat[:expected_input_dim]
            
            boundary_scores[i] = self.boundary_encoder(combined_feat).squeeze()
        return boundary_scores

    def _compute_separation_confidence(self,
                                       points: torch.Tensor,
                                       boundary_prob: torch.Tensor,
                                       leaf_mask: torch.Tensor) -> torch.Tensor:
        if torch.sum(leaf_mask) == 0:
            return torch.tensor(0.0, device=points.device)
        boundary_clarity = torch.var(boundary_prob[leaf_mask.bool()])
        boundary_points = points[boundary_prob > 0.7]
        if len(boundary_points) > 5:
            boundary_continuity = self._measure_boundary_continuity(boundary_points)
        else:
            boundary_continuity = torch.tensor(0.0, device=points.device)
        confidence = boundary_clarity * boundary_continuity
        return torch.clamp(confidence, 0, 1)

    def _measure_boundary_continuity(self, boundary_points: torch.Tensor) -> torch.Tensor:
        if len(boundary_points) < 3:
            return torch.tensor(0.0, device=boundary_points.device)
        distances = torch.norm(boundary_points[1:] - boundary_points[:-1], dim=1)
        distance_variance = torch.var(distances)
        continuity = 1.0 / (distance_variance + 1e-8)
        return torch.clamp(continuity, 0, 1)


class LeafSeparationNetwork(nn.Module):
    def __init__(self, feature_dim: int, k: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.separation_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )
        self.boundary_enhancer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def forward(self,
                points: torch.Tensor,
                features: torch.Tensor,
                boundary_info: Dict,
                leaf_mask: torch.Tensor) -> torch.Tensor:
        boundary_prob = boundary_info['boundary_prob']
        enhanced_features = self._enhance_boundary_differences(features, boundary_prob, leaf_mask, points)
        separated_features = self._propagate_separation(points, enhanced_features, boundary_prob, leaf_mask)
        return separated_features

    def _enhance_boundary_differences(self,
                                      features: torch.Tensor,
                                      boundary_prob: torch.Tensor,
                                      leaf_mask: torch.Tensor,
                                      points: torch.Tensor) -> torch.Tensor:
        batch_size, num_points = features.shape[:2]
        enhanced_features = features.clone()
        for b in range(batch_size):
            leaf_indices = torch.where(leaf_mask[b] > 0)[0]
            if len(leaf_indices) < 10:
                continue
            for i in leaf_indices:
                if boundary_prob[b, i] > 0.5:
                    center_feat = features[b, i]
                    distances = torch.norm(points[b] - points[b, i], dim=1)
                    _, neighbor_indices = torch.topk(distances, min(self.k, num_points), largest=False)
                    neighbor_feats = features[b, neighbor_indices]
                    mean_neighbor_feat = torch.mean(neighbor_feats, dim=0)
                    difference = center_feat - mean_neighbor_feat
                    enhanced_diff = difference * (1 + boundary_prob[b, i])
                    enhanced_features[b, i] = mean_neighbor_feat + enhanced_diff
        return enhanced_features

    def _propagate_separation(self,
                               points: torch.Tensor,
                               features: torch.Tensor,
                               boundary_prob: torch.Tensor,
                               leaf_mask: torch.Tensor) -> torch.Tensor:
        return features


class LeafCompletionNetwork(nn.Module):
    def __init__(self, feature_dim: int, k: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.completion_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

    def forward(self,
                points: torch.Tensor,
                features: torch.Tensor,
                leaf_mask: torch.Tensor,
                stage_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = points.shape[0]
        completed_features = []
        for b in range(batch_size):
            completed_feat = self._complete_leaf_gaps(points[b], features[b], leaf_mask[b])
            completed_features.append(completed_feat)
        return torch.stack(completed_features, dim=0)

    def _complete_leaf_gaps(self,
                            points: torch.Tensor,
                            features: torch.Tensor,
                            leaf_mask: torch.Tensor) -> torch.Tensor:
        completed_features = features.clone()
        if torch.sum(leaf_mask) < 10:
            return completed_features
        gap_regions = self._detect_leaf_gaps(points, leaf_mask)
        for gap_center, gap_radius in gap_regions:
            self._fill_gap_region(points, completed_features, gap_center, gap_radius, leaf_mask)
        return completed_features

    def _detect_leaf_gaps(self,
                           points: torch.Tensor,
                           leaf_mask: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        gaps = []
        if torch.sum(leaf_mask) < 10:
            return gaps
        leaf_points = points[leaf_mask.bool()]
        for i in range(0, len(leaf_points), 10):
            center = leaf_points[i]
            distances = torch.norm(points - center, dim=1)
            nearby_leaf_points = torch.sum((distances < 0.05) & leaf_mask.bool())
            if nearby_leaf_points < 3:
                gaps.append((center, 0.05))
        return gaps

    def _fill_gap_region(self,
                         points: torch.Tensor,
                         features: torch.Tensor,
                         gap_center: torch.Tensor,
                         gap_radius: float,
                         leaf_mask: torch.Tensor):
        distances = torch.norm(points - gap_center, dim=1)
        nearby_mask = (distances > gap_radius) & (distances < gap_radius * 2) & leaf_mask.bool()
        if torch.sum(nearby_mask) > 3:
            nearby_features = features[nearby_mask]
            avg_feature = torch.mean(nearby_features, dim=0)
            gap_mask = distances <= gap_radius
            features[gap_mask] = avg_feature


class InstanceEmbeddingNetwork(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    def forward(self, features: torch.Tensor, leaf_mask: torch.Tensor) -> torch.Tensor:
        # 检查特征张量的形状
        if features.dim() != 2:
            features = features.view(features.shape[0], -1)
        
        # 如果特征维度不匹配，进行线性变换
        if features.shape[-1] != self.embedding_net[0].in_features:
            if not hasattr(self, 'feature_proj'):
                self.feature_proj = nn.Linear(features.shape[-1], self.embedding_net[0].in_features).to(features.device)
            features = self.feature_proj(features)
        
        embeddings = self.embedding_net(features)
        embeddings = embeddings * leaf_mask.unsqueeze(-1)
        return embeddings


class InstanceConsistencyNetwork(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.consistency_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    def forward(self,
                points: torch.Tensor,
                embeddings: torch.Tensor,
                leaf_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_points = points.shape[:2]
        refined_embeddings = []
        for b in range(batch_size):
            refined = self._refine_instance_consistency(points[b], embeddings[b], leaf_mask[b])
            refined_embeddings.append(refined)
        return torch.stack(refined_embeddings, dim=0)
    def _refine_instance_consistency(self,
                                     points: torch.Tensor,
                                     embeddings: torch.Tensor,
                                     leaf_mask: torch.Tensor) -> torch.Tensor:
        refined_embeddings = embeddings.clone()
        if torch.sum(leaf_mask) < 10:
            return refined_embeddings
        leaf_indices = torch.where(leaf_mask > 0)[0]
        for i in leaf_indices:
            center_embedding = embeddings[i]
            center_point = points[i]
            distances = torch.norm(points - center_point, dim=1)
            neighbor_mask = (distances < 0.03) & leaf_mask.bool()
            if torch.sum(neighbor_mask) > 1:
                neighbor_embeddings = embeddings[neighbor_mask]
                similarities = F.cosine_similarity(center_embedding.unsqueeze(0), neighbor_embeddings, dim=1)
                similar_neighbors = neighbor_embeddings[similarities > 0.7]
                if len(similar_neighbors) > 0:
                    mean_similar = torch.mean(similar_neighbors, dim=0)
                    combined = torch.cat([center_embedding, mean_similar])
                    refined_embeddings[i] = self.consistency_net(combined)
        return refined_embeddings


