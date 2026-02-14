"""
损失函数
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalTripletLoss(nn.Module):
    """
    Focal Triplet Loss，源自目标检测领域（RetinaNet）
    通过给难样本分配更大的权重，使模型更关注难以区分的样本。
    公式: Loss = (1 + basic_loss)^gamma * basic_loss
    """
    def __init__(self, margin: float = 0.3, gamma: float = 2.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        
    def forward(self, anchor, positive, negative):
        # 计算欧氏距离 (L2)
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)
        
        # 基础 Triplet Loss: max(d(a, p) - d(a, n) + margin, 0)
        basic_loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        
        # Focal Weighting: (1 + loss)^gamma
        # 当 loss 越大，权重呈指数级增长，强迫模型关注难例
        focal_loss = basic_loss * ((1 + basic_loss) ** self.gamma)
        
        return focal_loss.mean()

def create_loss_fn(margin: float):
    """
    创建损失函数。
    
    Args:
        margin (float): 正负样本对之间所需的最小距离。

    Returns:
        nn.Module: Loss 实例。
    """
    # --- Focal Triplet Loss (推荐用于提升 Hard Negative 挖掘能力) ---
    return FocalTripletLoss(margin=margin, gamma=2.0)

    # --- 标准 Triplet Loss  ---
    # return nn.TripletMarginLoss(margin=margin, p=2) 
